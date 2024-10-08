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

'''
import projects.nerf.utils.camera as camera
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
    parser.add_argument('--debugckpt', action='store_true')
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
    trainer.checkpointer.debugckpt = args.debugckpt
    trainer.debugckpt = args.debugckpt
    trainer.model.debugckpt = args.debugckpt
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
                

        # ----------- get 3 xyzabg cam and interp ----------------
        # cam_poses_render = []
        # cam_intr_render = []
        # vrep_Ms = []
        # scale=0.11426599033526427949695350
        # num_steps = [10]
        # time_step = 0.1
        # render_xyzabg = torch.tensor(np.loadtxt("pose_file/xyzabg.txt"), dtype=torch.float32).reshape(-1, 6)
        # for xyzabg in render_xyzabg:
        #     vrep_M = torch.eye(4) 
        #     vrep_M[:3, 3] = xyzabg[:3].T
        #     vrep_M[:3, :3] = XYZ_angle_to_rotation_matrix(xyzabg[3:])
        #     print("--------vrep_M--------")
        #     print(vrep_M)
        #     vrep_Ms.append(vrep_M)
        # assert len(num_steps) == len(vrep_Ms) - 1
        # P = camera.Pose()

        # for i in range(len(num_steps)):
        #     for t in range(num_steps[i]):
        #         cam_input = P.interpolate(vrep_Ms[i][:3], vrep_Ms[i+1][:3], t/num_steps[i])
        #         cam_input[:3, 3] *= 1/scale # 位移缩小scale倍
        #         cam_input[:2] *= -1
        #         cam_poses_render.append(cam_input[:3, :])
        #         print("----------interpolate_M----------")
        #         print(i, t, cam_input[:3, :])
        #         cam_intr_render.append(template_intr.clone())

        # cam_poses_vrep = torch.stack(cam_poses_render, dim=0)
        # cam_intr_vrep = torch.stack(cam_intr_render, dim=0)
        # for i in range(cam_poses_vrep.shape[0]):
        #     infer_data = {"pose": cam_poses_vrep[i:(i+1)], "intr": cam_intr_vrep[i:(i+1)], "idx": torch.arange(1), "file_prefix": f"{i}"}
        #     trainer.inference_one_image(infer_data, reverse_xy=False)
            
            # every time take 8 images from cam_poses_vrep
            # import ipdb; ipdb.set_trace()
        # ----------- get interped xyzab and render---------------

        cam_poses_render = []
        cam_intr_render = []

        scale=0.11426599033526427949695350

        render_xyzabg = torch.tensor(np.loadtxt("pose_file/xyzabg.txt"), dtype=torch.float32).reshape(-1, 6)
        # import ipdb;ipdb.set_trace()
        for xyzabg in render_xyzabg:
            vrep_M = torch.eye(4) 
            vrep_M[:3, 3] = xyzabg[:3].T
            vrep_M[:3, :3] = XYZ_angle_to_rotation_matrix(xyzabg[3:])
            cam_input = vrep_M
            cam_input[:3, 3] *= 1/scale # 位移缩小scale倍
            cam_input[:2] *= -1
            cam_poses_render.append(cam_input[:3, :])
            cam_intr_render.append(template_intr.clone())

        cam_poses_vrep = torch.stack(cam_poses_render, dim=0)
        cam_intr_vrep = torch.stack(cam_intr_render, dim=0)
        for i in range(cam_poses_vrep.shape[0]):
            infer_data = {"pose": cam_poses_vrep[i:(i+1)], "intr": cam_intr_vrep[i:(i+1)], "idx": torch.arange(1), "file_prefix": f"{i}"}
            trainer.inference_one_image(infer_data, reverse_xy=False)
        return
        # infer_data = {"pose": cam_poses_vrep[i:(i*8)], "intr": cam_intr_vrep[i:(i*8)], "idx": torch.arange(cam_poses_vrep[i:(i*8)].shape[0]), 
        #             "file_prefix": [j for j in range(8)]}
        # trainer.inference_one_image(infer_data, reverse_xy=False)
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

