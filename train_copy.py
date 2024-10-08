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
物体操作是好的 相机操作三个角组合起来不行
'''

import argparse
import os
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
def euler_from_matrix(matrix):
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

def XYZ_angle_to_rotation_matrix(a, ZYX=False):  # a radius   axis X Y Z
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
    trainer.set_data_loader(cfg, split="test")
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
        W, H = (533, 300)
        trainer.current_epoch = trainer.checkpointer.resume_epoch or trainer.current_epoch
        trainer.current_iteration = trainer.checkpointer.resume_iteration or trainer.current_iteration
        trainer.mode = 'val'
        template_data = torch.load("eval_data.pt")
        template_intr = template_data["intr"][0]
        print("intr is ", template_intr)
        FOV_W = 2 * math.atan(0.5 * W / template_intr[0, 0])
        FOV_H = 2 * math.atan(0.5 * W / template_intr[1, 1])
        print("FOV_W, FOV_H", FOV_W * 180 / np.pi, FOV_H * 180 / np.pi)

        template_pose = template_data["pose"][0]
        template_intr[0] *= W / template_data['image'].shape[3] 
        template_intr[1] *= H / template_data['image'].shape[2] 
        infer_poses = []
        infer_intr = []
        # ------------------align 来自于meshlab，但第一行第三行的前三列乘了负号------------------
        align_mat = torch.tensor(np.loadtxt("pose_file/align.txt"), dtype=torch.float32)[:4]
        scale = torch.norm(align_mat[:3, 0])
        print("scale is", scale)
        align_mat /= scale
        # only for meshlab
        align_mat[3, 3] = 1
        # align_mat[:3, 3] = 0
        align_XYZ_angle = euler_from_matrix(torch.inverse(align_mat[:3, :3]))
        align_mat[:3, :3] = XYZ_angle_to_rotation_matrix(align_XYZ_angle, ZYX=True)
        print("euler angle:", euler_from_matrix(torch.inverse(align_mat[:3, :3])) * 180 / np.pi)

        txt_poses = torch.tensor(np.loadtxt("pose_file/inference_poses.txt"), dtype=torch.float32)
        txt_poses = txt_poses.reshape(-1, 4, 4)

        cam_poses_txt = torch.tensor(np.loadtxt("pose_file/cam_poses.txt"), dtype=torch.float32).reshape(-1, 6) * -1
        # cam_intrs_txt = torch.tensor(np.loadtxt("pose_file/cam_intrs.txt"), dtype=torch.float32).reshape(-1, 6)
        obj_poses_txt = torch.tensor(np.loadtxt("pose_file/obj_poses.txt"), dtype=torch.float32).reshape(-1, 6) * -1
        obj_poses_txt = obj_poses_txt[:cam_poses_txt.shape[0]]

        # import ipdb; ipdb.set_trace()
        cam_poses_render = []
        cam_intr_render = []
        for i in range(len(cam_poses_txt)):
            M = torch.eye(4) # camera to world
            M[:3, 3] = cam_poses_txt[i][:3].T
            M[:3, :3] = XYZ_angle_to_rotation_matrix(cam_poses_txt[i][3:] / 180 * np.pi, ZYX=True)
            # M *= -1

            M2 = torch.eye(4) 
            M2[:3, 3] = obj_poses_txt[i][:3].T
            M2[:3, :3] = XYZ_angle_to_rotation_matrix(obj_poses_txt[i][3:] / 180 * np.pi, ZYX=True) 

            M3 = torch.inverse(M2) @ torch.inverse(align_mat) # object to world
            M4 = M @ M3 #
            # print(f"NO ---- {i}", "M\n",M,"\nM2\n", M2, " \nM@M2^-1\n", M @ torch.inverse(M2))
            cam_poses_render.append(M4[:3, :])
            # cam_poses_render.append(M[:3, :])
            # cam_intr_render.append(cam_intrs_txt[i])
            cam_intr_render.append(template_intr.clone())
            # Xo=Mco Xc= Mcw Mwo Xc
        cam_poses_vrep = torch.stack(cam_poses_render, dim=0)
        cam_intr_vrep = torch.stack(cam_intr_render, dim=0)
        infer_data = {"pose": cam_poses_vrep, "intr": cam_intr_vrep, "idx": torch.arange(cam_poses_vrep.shape[0])}

        # nerf
        # 世界坐标系xyz 从Z负方向看，分别是左上外
        

        # 为什么负角度反而是顺时针呢?(毕竟是左手坐标系)
    

        # z = np.array([0, np.cos(45 / 180 * np.pi), np.sin(45 / 180 * np.pi)]).T
        # x = np.array([1, 0, 0]).T
        # y = np.cross(z, x)
        # txt_poses[0][:3,:3] = torch.tensor([x, y, z])

        # txt_poses = txt_poses[0:3, ...]
        # for i in range(txt_poses.shape[0]):
        #     infer_poses.append(txt_poses[i, :3, :])
        #     infer_intr.append(template_intr.clone())

        # for i in range(txt_poses.shape[0]): # right
        #     infer_poses.append((txt_poses[i] @ torch.inverse(align_mat))[:3, :])
        #     infer_intr.append(template_intr.clone())
        # infer_poses = torch.stack(infer_poses, dim=0)
        # infer_intr = torch.stack(infer_intr, dim=0)
        # infer_data = {"pose": infer_poses, "intr": infer_intr, "idx": torch.arange(infer_poses.shape[0])}

        
        trainer.inference_one_image(infer_data, reverse_xy=False)
        
        # TODO: complete following code
        # import ipdb; ipdb.set_trace()
        # render_one_image(trainer, cam_poses_vrep, cam_intr_vrep)
        

        
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

