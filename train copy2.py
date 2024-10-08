'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------\
###################
物体操作是好的 相机操作不行
'''

import argparse
import os, ipdb
import torch, math
import numpy as np
import imaginaire.config
from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.distributed import init_dist, get_world_size, master_only_print as print, is_master
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.trainers.utils.logging import init_logging
from imaginaire.trainers.utils.get_trainer import get_trainer
from imaginaire.utils.set_random_seed import set_random_seed
def clean_folder(folder):
    import os

    # specify the path of the folder to clean
    folder_path = folder

    # remove all files and subdirectories within the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
def ZYX_matrix_to_euler(matrix): #radius
    '''
    Extract the eular angle from a rotation matrix
    '''
    _EPS = 1e-8
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])
    ay = math.atan2(-M[2, 0], cy)
    if ay < -math.pi / 2 + _EPS and ay > -math.pi / 2 - _EPS:  # pitch = -90 deg
        ax = 0
        az = math.atan2(-M[1, 2], -M[0, 2])
    elif ay < math.pi / 2 + _EPS and ay > math.pi / 2 - _EPS:
        ax = 0
        az = math.atan2(M[1, 2], M[0, 2])
    else:
        ax = math.atan2(M[2, 1], M[2, 2])
        az = math.atan2(M[1, 0], M[0, 0])

    # ay = math.asin(M[2, 0])
    # ax = math.atan2(M[2, 1], M[2, 2])
    # az = math.atan2(M[1, 0], M[0, 0])
    return np.array([ax, ay, az])
def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', help='Path to the training config file.', required=True)
    parser.add_argument('--logdir', help='Dir for saving logs and models.', default=None)
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--show_pbar', action='store_true')
    parser.add_argument('--wandb', action='store_true', help="Enable using Weights & Biases as the logger")
    parser.add_argument('--wandb_name', default='default', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--inference', action='store_true')
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd

def get_cam_pointing_p(cam_pos=[0.0, 0.0, 5.0], point=[0.0, 0.0, 0.0]):
    camera_position = torch.tensor([cam_pos])
    camera_direction = torch.tensor([point]) - camera_position  # point camera towards the origin
    camera_right = torch.cross(torch.tensor([[0.0, 1.0, 0.0]]), camera_direction)

    camera_direction = camera_direction / torch.norm(camera_direction, dim=1, keepdim=True)  # normalize direction vector
    camera_right = camera_right / torch.norm(camera_right, dim=1, keepdim=True)  # normalize right vector
    camera_up = torch.cross(camera_direction, camera_right)

    # Compute the camera pose
    R = torch.stack([camera_right[0], camera_up[0], camera_direction[0]], dim=1)
    # T = -torch.matmul(R, camera_position.transpose(0, 1)).squeeze()
    infer_pose = torch.eye(4, dtype=torch.float32)

    infer_pose[:3, :3] = R
    # import ipdb; ipdb.set_trace()
    infer_pose[:3, 3] = camera_position[0]
    # infer_pose = infer_pose[:3, :]
    return infer_pose

def angle_to_rotation_matrix(a, axis):  # a radius   axis X Y Z
    # Get the rotation matrix from Euler angle around specific axis.
    roll = dict(X=1, Y=2, Z=0)[axis]
    if isinstance(a, float):
        a = torch.tensor(a)
    zero = torch.zeros_like(a)
    eye = torch.ones_like(a)
    M = torch.stack([torch.stack([a.cos(), -a.sin(), zero], dim=-1),
                     torch.stack([a.sin(), a.cos(), zero], dim=-1),
                     torch.stack([zero, zero, eye], dim=-1)], dim=-2)
    M = M.roll((roll, roll), dims=(-2, -1))
    return M

def XYZ_angle_to_rotation_matrix(a, ZYX=False): 
    if ZYX:
        return angle_to_rotation_matrix(a[2], "Z") @ angle_to_rotation_matrix(a[1], "Y") @ angle_to_rotation_matrix(a[0], "X")
    else:
        # import ipdb; ipdb.set_trace()
        return angle_to_rotation_matrix(a[0], "X") @ angle_to_rotation_matrix(a[1], "Y") @ angle_to_rotation_matrix(a[2], "Z")

def render_one_image(trainer, cam_poses_vrep, cam_intr_vrep, reverse_xy=False):
        infer_data = {"pose": cam_poses_vrep, "intr": cam_intr_vrep, "idx": torch.arange(cam_poses_vrep.shape[0])}
        # infer_data = {"pose": infer_poses, "intr": infer_intr, "idx": torch.arange(infer_poses.shape[0])}
        trainer.inference_one_image(infer_data, reverse_xy=reverse_xy)

def main():
    args, cfg_cmd = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)
    if args.inference:
        cfg.data.val.image_size = cfg.data.test.image_size
    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)

    # If args.single_gpu is set to True, we will disable distributed data parallel.
    if not args.single_gpu:
        # this disables nccl timeout
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Training with {get_world_size()} GPUs.")

    # set random seed by rank
    set_random_seed(args.seed, by_rank=True)

    # Global arguments.
    imaginaire.config.DEBUG = args.debug

    # Create log directory for storing training results.
    cfg.logdir = init_logging(args.config, args.logdir, makedir=True)

    # Print and save final config
    if is_master():
        cfg.print_config()
        cfg.save_config(cfg.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.

    trainer = get_trainer(cfg, is_inference=args.inference or args.test, seed=args.seed) # FIXME: not sure if  is is_inference=args.inference
    # import ipdb; ipdb.set_trace()
    trainer.set_data_loader(cfg, split="train")
    trainer.set_data_loader(cfg, split="val")
    # trainer.set_data_loader(cfg, split="test")
    trainer.checkpointer.load(args.checkpoint, args.resume, load_sch=True, load_opt=True)
    
    # Initialize Wandb.
    trainer.init_wandb(cfg,
                       project=args.wandb_name,
                       mode="disabled" if args.debug or not args.wandb else "online",
                       resume=args.resume,
                       use_group=True)

    trainer.mode = 'train'
    # Start training.
    if args.inference:
        clean_folder("outdir")
        # W, H = (533, 300)
        # W, H = (300, 300)
        H, W = cfg.data.test.image_size # check check
        trainer.current_epoch = trainer.checkpointer.resume_epoch or trainer.current_epoch
        trainer.current_iteration = trainer.checkpointer.resume_iteration or trainer.current_iteration
        trainer.mode = 'val'
        template_data = torch.load("eval_data.pt")
        template_intr = template_data["intr"][0]

        template_pose = template_data["pose"][0]
        template_intr[0, 2] *= W / template_data['image'].shape[3]
        template_intr[1, 2] *= H / template_data['image'].shape[2]

        print("intr is ", template_intr)
        FOV_W = 2 * math.atan(0.5 * W / template_intr[0, 0])
        FOV_H = 2 * math.atan(0.5 * H / template_intr[1, 1])
        print("FOV_W, FOV_H", FOV_W * 180 / np.pi, FOV_H * 180 / np.pi)
        # ipdb.set_trace()
        # ------------------align 来自于meshlab，但第一行第三行的前三列乘了负号------------------
        # euler_from_matrix(XYZ_angle_to_rotation_matrix(align_XYZ_angle / 180 * np.pi, ZYX=True))*180/np.pi - align_XYZ_angle.cpu().numpy()
        # import ipdb; ipdb.set_trace()
        obj_vrep_poses = torch.tensor(np.loadtxt("pose_file/inference_poses.txt"), dtype=torch.float32)
        obj_vrep_poses = obj_vrep_poses.reshape(-1, 4, 4)
        txt_cam_poses = torch.tensor(np.loadtxt("pose_file/cam_poses.txt"), dtype=torch.float32).reshape(-1, 6)
        # cam_intrs_txt = torch.tensor(np.loadtxt("pose_file/cam_intrs.txt"), dtype=torch.float32).reshape(-1, 6)
        txt_obj_poses = torch.tensor(np.loadtxt("pose_file/obj_poses.txt"), dtype=torch.float32).reshape(-1, 6)
        txt_obj_poses = txt_obj_poses[:txt_cam_poses.shape[0]]

        # # import ipdb; ipdb.set_trace()
        # cam_poses_render = []
        # cam_intr_render = []

        # # 先都用VREP里的相机定义，假设物体未对齐
        # # 先对齐VREP的物体（物体外参变一次）
        # # 然后再利用现在物体坐标对物体进行位姿变换
        # # 求出VREP里相机相对物体(vrep)的pose
        # # 然后再把相对pose用inverse_M变换到物体（nerf）当中。然后再对z轴取反

        # for i in range(len(txt_cam_poses)):
        # # import ipdb; ipdb.set_trace()
        # # for i in range(5):
        #     M_obj = torch.eye(4) # object to world_vrep
        #     # M_obj = align_mat @ M_obj

        #     M_obj_vrep = torch.eye(4) 
        #     M_obj_vrep[:3, 3] = txt_obj_poses[i][:3].T
        #     M_obj_vrep[:3, :3] = XYZ_angle_to_rotation_matrix(txt_obj_poses[i][3:] / 180 * np.pi)#, ZYX=True) 
        #     M_obj = M_obj_vrep @ M_obj


        #     M_cam = torch.eye(4) # camera to world_vrep
        #     M_cam[:3, 3] = txt_cam_poses[i][:3].T
        #     M_cam[:3, :3] = XYZ_angle_to_rotation_matrix(txt_cam_poses[i][3:] / 180 * np.pi)#, ZYX=True)

        #     M_cam_obj = M_cam @ torch.inverse(M_obj) # camera to object(vrep)
            
        #     M_obj_vrep_to_obj_nerf = torch.eye(4) * -1
        #     M_obj_vrep_to_obj_nerf[3, 3] = 1

        #     # M_cam_obj_nerf = M_cam_obj @ M_obj_vrep_to_obj_nerf # camera to object(nerf) Mcv * Mvn = Mcn 为什么是左乘？？？？ 
        #     M_cam_obj_nerf = M_obj_vrep_to_obj_nerf @ M_cam_obj
        #     M_cam_obj_nerf[:3, 2] *= -1 # z轴取反 因为相机定义不同
        #     # -1 -1 1 -5
        #     # 1 1 1 5
        #     # print(M_cam_obj, "\n", M_cam_obj_nerf)
        #     # import ipdb; ipdb.set_trace()
        #     cam_poses_render.append(M_cam_obj_nerf[:3, :])
        #     cam_intr_render.append(template_intr.clone())
        #     # Xo=Mco Xc= Mcw Mwo Xc
        # cam_poses_vrep = torch.stack(cam_poses_render, dim=0)
        # cam_intr_vrep = torch.stack(cam_intr_render, dim=0)
        # infer_data = {"pose": cam_poses_vrep, "intr": cam_intr_vrep, "idx": torch.arange(cam_poses_vrep.shape[0])}
        
        # -----------------手动调nerf渲染--------------
        # # a=15 物体绕x顺时针15度
        # # a=15 M=M^-1 坏掉，感觉是某个背面 
        # # a=15 M[:3, 3] = torch.inverse(M)[:3, 3] 坏掉，感觉是某个背面 
        # # a=-15 M[:3, 3] = -1 * torch.inverse(M)[:3, 3]
        # while True:
        #     abg = [15, 0, 0]
        #     M = torch.eye(4) 
        #     M[2, 3] = 5
        #     M[:3, :3] = XYZ_angle_to_rotation_matrix(torch.tensor(abg) / 180 * np.pi)#, ZYX=True) 
        #     M[:3, 3] = -1 * torch.inverse(M)[:3, 3]
        #     cam_input = M[None, :3, ...]
        #     print(cam_input)
        #     infer_data = {"pose": cam_input, "intr": template_intr[None, ...], "idx": torch.arange(cam_input.shape[0])}
        #     trainer.inference_one_image(infer_data, reverse_xy=False)
        #     ipdb.set_trace()

        # ----------- manual infer ----------------
        scale=0.11426599033526427949695350
        # scale=1
        for i in [1]:
            for j in [1/scale]:
                obj2cam = torch.tensor([-0.8820955172, 0.4330189141, -0.1854780814, -0.01305898257, 
                                        -0.4160712195, -0.9008007187, -0.1242690849, 0.1207411696, 
                                        -0.2208896532, -0.03244511117, 0.9747589835, 0.1663401123]).reshape(3,4)
                obj2cam = torch.tensor([-0.9061894834, 0.3891216911, -0.1655443434, 0.022, -0.4174347219, -0.7605682082, 0.497276838, -0.028, 0.06759343956, 0.5197309979, 0.8516518166, 0.41]).reshape(3,4)
                obj2cam = torch.tensor([-0.02305567159, 0.7578943937, -0.6519697263, 0.04361646502, -0.4758621395, -0.5818476768, -0.6595517456, -0.01188085486, -0.879217641, 0.2950413004, 0.3740681367, 0.3422986851]).reshape(3,4)
                # obj2cam = torch.tensor([-0.2161168788, -0.858681947, 0.4647136846, 3.428711689, 0.3184585036, -0.5119262439, -0.7978193419, -0.9878645724, 0.9229721969, -0.02443020139, 0.3840904698, 6.799547669]).reshape(3,4)
                cam_intr = template_intr.clone()
                cam_input = obj2cam
                # scale = 0.1142
                # cam_input /= scale
                # import ipdb; ipdb.set_trace()
                cam_input[:3, :3] *= i # 模型缩小scale倍
                cam_input[:3, 3] *= j # 位移缩小scale倍
                cam_input[:2] *= -1
                cam_input = cam_input[None, ...]
                print(cam_input)
                infer_data = {"pose": cam_input, "intr": cam_intr[None, ...], "idx": torch.arange(cam_input.shape[0]), "file_prefix": f"cam2"}
                trainer.inference_one_image(infer_data, reverse_xy=False)
        return
    
        # for set_T in [True, False]:

        #     cam_intr = template_intr.clone()
        #     cam_input = obj2cam
        #     # scale = 0.1142
        #     # cam_input /= scale
        #     # import ipdb; ipdb.set_trace()
        #     cam_input[:3, :3] *= i # 模型缩小scale倍
        #     cam_input[:3, 3] *= j # 位移缩小scale倍
        #     cam_input[:2] *= -1
        #     cam_input = cam_input[None, ...]
        #     print(cam_input)
        #     infer_data = {"pose": cam_input, "intr": cam_intr[None, ...], "idx": torch.arange(cam_input.shape[0]), "file_prefix": f"set_T{set_T}"}
        #     trainer.inference_one_image(infer_data, reverse_xy=False)
        # # break
        # return 
    
        # ---------------------------
        align_params = torch.tensor(np.loadtxt("pose_file/align.txt"), dtype=torch.float32).reshape(-1, 6)
        txt_cam_poses = torch.tensor(np.loadtxt("pose_file/cam_poses.txt"), dtype=torch.float32).reshape(-1, 6)
        txt_obj_poses = torch.tensor(np.loadtxt("pose_file/obj_poses.txt"), dtype=torch.float32).reshape(-1, 6)
        
        txt_cam_poses[:, 5] += 180 # invert z axis
        scale = 1
        template_intr[0, 0] *= scale
        template_intr[1, 1] *= scale


        align_mat_R = torch.eye(4)
        align_mat_R[:3, :3] = XYZ_angle_to_rotation_matrix(align_params[0][3:] / 180 * np.pi)
        align_mat_T = align_params[0][:3].T
        # print("align_mat", align_mat)

        vrep_obj_pose_init = torch.diag(torch.tensor([-1.0, -1.0, -1.0, 1.0])) # vrep 可视化和帮忙计算也行
        vrep_obj_pose_init[:3, 3] += align_mat_T

        vrep_cam_pose_init = torch.diag(torch.tensor([-1.0, -1.0, -1.0, 1.0])) 
        vrep_cam_pose_init = align_mat_R @ vrep_cam_pose_init 

        cam_poses_render = []
        cam_intr_render = []
        for i in range(len(txt_cam_poses)):
            
            obj_abg = txt_cam_poses[i][3:]
            cam_abg = txt_obj_poses[i][3:]
            obj_trans = txt_obj_poses[i][:3]
            cam_trans = txt_cam_poses[i][:3]

            vrep_obj_pose = torch.eye(4)
            vrep_obj_pose[:3, 3] = obj_trans.T 
            vrep_obj_pose[:3, :3] = XYZ_angle_to_rotation_matrix(obj_abg / 180 * np.pi)
            vrep_obj_pose = vrep_obj_pose @ vrep_obj_pose_init

            vrep_cam_pose = torch.eye(4)
            vrep_cam_pose[:3, :3] = XYZ_angle_to_rotation_matrix(cam_abg / 180 * np.pi)
            vrep_cam_pose[:3, 3] = cam_trans.T
            vrep_cam_pose = vrep_cam_pose @ vrep_cam_pose_init

            cam_to_obj = torch.inverse(vrep_obj_pose) @ vrep_cam_pose 
            cam_input = cam_to_obj[:3, :]
            cam_poses_render.append(cam_input)
            cam_intr_render.append(template_intr.clone())
            print(cam_input)

            # cam_input[:, :3, 3] = 0
        cam_poses_vrep = torch.stack(cam_poses_render, dim=0)
        cam_intr_vrep = torch.stack(cam_intr_render, dim=0)
        # import ipdb; ipdb.set_trace()
        infer_data = {"pose": cam_poses_vrep, "intr": cam_intr_vrep, "idx": torch.arange(cam_poses_vrep.shape[0])}
        trainer.inference_one_image(infer_data, reverse_xy=False)


        # ----------------backup---------------------
        # file_prefix = ["ct111"]

        # abg = torch.tensor([127.0, 30.0, 120.7]) # 11,-30,-61
        # abg = torch.tensor([0, 0, 0]) 
        # obj_trans = torch.tensor([1.0, 1.0, 1.0]) # debug

        # cam_abg = torch.tensor([0, 0, 0]) 
        # cam_trans = torch.tensor([0.0, 0.0, -5.0]) # debug
        # cam_pose_init = torch.eye(4) * -1
        # cam_pose_init[3,3] = 1

        # vrep_obj_pose_init=torch.diag(torch.tensor([-1.0, -1.0, -1.0, 1.0])) # vrep 可视化和帮忙计算也行
        # vrep_cam_pose_init=torch.diag(torch.tensor([-1.0, -1.0, -1.0, 1.0])) 

        # while True:
        #     vrep_obj_pose = torch.eye(4)
        #     vrep_obj_pose[:3, 3] = obj_trans.T
        #     vrep_obj_pose[:3, :3] = XYZ_angle_to_rotation_matrix(abg / 180 * np.pi)
        #     vrep_obj_pose = vrep_obj_pose @ vrep_obj_pose_init

        #     vrep_cam_pose = torch.eye(4)
        #     vrep_cam_pose[:3, :3] = XYZ_angle_to_rotation_matrix(cam_abg / 180 * np.pi)
        #     vrep_cam_pose[:3, 3] = cam_trans.T
        #     vrep_cam_pose = vrep_cam_pose @ vrep_cam_pose_init

        #     vrep_to_nerf_coord = torch.eye(4) * -1
        #     vrep_to_nerf_coord[3, 3] = 1
        #     # nerf_obj_pose = vrep_obj_pose @ torch.inverse(vrep_to_nerf_coord) # 被动坐标变换 存疑 wrong

        #     # nerf_obj_pose = vrep_to_nerf_coord @ vrep_obj_pose 
        #     # cam_to_nerf_obj = torch.inverse(nerf_obj_pose) @ cam_pose
        #     # cam_input = cam_to_nerf_obj[None, :3, :]

        #     cam_to_obj = torch.inverse(vrep_obj_pose) @ vrep_cam_pose 
        #     # cam_to_obj = vrep_to_nerf_coord @ cam_to_obj
        #     # cam_input = torch.inverse(cam_to_obj)
        #     cam_input = cam_to_obj[None, :3, :]

        #     # cam_input[:, :3, 3] = 0
        #     infer_data = {"pose": cam_input, "intr": infer_intr[:1], "idx": torch.arange(1), "file_prefix": file_prefix}
        #     trainer.inference_one_image(infer_data, reverse_xy=False)
        #     import ipdb; ipdb.set_trace()
        
    if args.test:
        # cfg.max_epoch = 1
        # cfg.max_iter = 1
        trainer.current_epoch = trainer.checkpointer.resume_epoch or trainer.current_epoch
        trainer.current_iteration = trainer.checkpointer.resume_iteration or trainer.current_iteration
        data_all = trainer.test(trainer.test_data_loader, mode="test", show_pbar=True)

    if not args.test and not args.inference:
        trainer.train(cfg,
                    trainer.train_data_loader,
                    single_gpu=args.single_gpu,
                    profile=args.profile,
                    show_pbar=args.show_pbar)

    # Finalize training.
    trainer.finalize(cfg)


if __name__ == "__main__":
    main()

