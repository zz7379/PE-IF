import os
import torch
import imageio
import numpy as np
import skimage
import cv2
from inerf_utils import find_POI, img2mse, to8b

import torch.nn as nn
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def vec2ss_matrix(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix

class camera_transf(nn.Module):
    def __init__(self):
        super(camera_transf, self).__init__()
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.theta = nn.Parameter(torch.normal(0., 1e-6, size=()))

    def forward(self, x):
        exp_i = torch.zeros((4,4)).to(self.w.device)
        w_skewsym = vec2ss_matrix(self.w).to(self.w.device)
        exp_i[:3, :3] = torch.eye(3, device=self.w.device) + torch.sin(self.theta) * w_skewsym + (1 - torch.cos(self.theta)) * torch.matmul(w_skewsym, w_skewsym)
        exp_i[:3, 3] = torch.matmul(torch.eye(3, device=self.w.device) * self.theta + (1 - torch.cos(self.theta)) * w_skewsym + (self.theta - torch.sin(self.theta)) * torch.matmul(w_skewsym, w_skewsym), self.v)
        exp_i[3, 3] = 1.
        T_i = torch.matmul(exp_i, x)
        return T_i




DEBUG = False
OVERLAY = True

if __name__=='__main__':

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args, cfg_cmd = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)
    if args.inference:
        cfg.data.val.image_size = cfg.data.test.image_size
    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)
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
    trainer.set_data_loader(cfg, split="train")
    trainer.set_data_loader(cfg, split="val")
    # trainer.set_data_loader(cfg, split="test")
    if not args.debugckpt:
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
        trainer.debugckpt = args.debugckpt
        if not args.debugckpt:
            trainer.current_epoch = trainer.checkpointer.resume_epoch or trainer.current_epoch
            trainer.current_iteration = trainer.checkpointer.resume_iteration or trainer.current_iteration
        else:
            trainer.current_epoch = 0
            trainer.current_iteration = 0
            trainer.checkpointer.debugckpt = True

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

        scale=0.11426599033526427949695350

        # --------------- nerf ready ----------------
        xyzabg = torch.tensor(np.loadtxt("pose_file/inerf_xyzabg.txt"), dtype=torch.float32).reshape(-1, 6)
        obs_img = cv2.imread( "pose_file/Image1.png")
        sampling_strategy = "random"
        vrep_mask = cv2.imread( "pose_file/vrep_mask.png")
        lrate = 0.01
        inerf_batch_size = 512
        dil_iter = 3
        # -------------------------------------------

        obs_img = (np.array(obs_img) / 255.).astype(np.float32)
        
        pose = torch.eye(4)
        pose[:3, :3] = XYZ_angle_to_rotation_matrix(xyzabg[0, 3:])
        pose[:3, 3] = xyzabg[0, :3]
        cam_poses_render = pose[None, :3, :]
        cam_intr_render = template_intr.clone()[None, ...]
        infer_data = {"pose": cam_poses_render, "intr": cam_intr_render, "idx": torch.arange(1), "file_prefix": f"{0}"}
        # trainer.inference_one_image(infer_data, reverse_xy=False)

        start_pose = pose.clone().detach()
        
        # downsample obs_img to mask size
        obs_img = cv2.resize(obs_img, (vrep_mask.shape[1], vrep_mask.shape[0]), interpolation=cv2.INTER_AREA)
        if type(vrep_mask) != type(None):
            # raise NotImplementedError
            dilated_vrep_mask = cv2.dilate(vrep_mask, np.ones((20, 20), np.uint8), iterations=3)
            cv2.imwrite("./pose_file/dilated_vrep_mask.png", dilated_vrep_mask)
            obs_img_masked = obs_img * dilated_vrep_mask
            cv2.imwrite("./pose_file/obs_img_masked.png", obs_img_masked)
            # 
        else:
            obs_img_masked = obs_img     
        
        # turn obs_img_masked to grayscale
        obs_img_masked = (np.array(obs_img_masked) * 255).astype(np.uint8)
        POI = find_POI(obs_img_masked, DEBUG)  # xy pixel coordinates of points of interest (N x 2) using SIFT
        obs_img_masked = (np.array(obs_img_masked) / 255.).astype(np.float32)
        

        # create meshgrid from the observed image
        coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H)), -1),
                            dtype=int)

        # create sampling mask for interest region sampling strategy
        interest_regions = np.zeros((H, W, ), dtype=np.uint8)
        interest_regions[POI[:,1], POI[:,0]] = 1
        interest_regions = cv2.dilate(interest_regions, np.ones((5, 5), np.uint8), iterations=dil_iter)
        interest_regions = np.array(interest_regions, dtype=bool)
        interest_regions = coords[interest_regions]

        # not_POI -> contains all points except of POI
        coords = coords.reshape(H * W, 2)
        not_POI = set(tuple(point) for point in coords) - set(tuple(point) for point in POI)
        not_POI = np.array([list(point) for point in not_POI]).astype(int)

        # Create pose transformation model
        start_pose = torch.Tensor(start_pose).to(device)
        cam_transf = camera_transf().to(device)
        optimizer = torch.optim.Adam(params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999))
        
        os.makedirs("./inerf_output", exist_ok=True)

        # imgs - array with images are used to create a video of optimization process
        if OVERLAY is True:
            imgs = []

        for k in range(600):
            if sampling_strategy == 'random':
                rand_inds = np.random.choice(coords.shape[0], size=inerf_batch_size, replace=False)
                batch = coords[rand_inds]

            elif sampling_strategy == 'interest_points':
                if POI.shape[0] >= inerf_batch_size:
                    rand_inds = np.random.choice(POI.shape[0], size=inerf_batch_size, replace=False)
                    batch = POI[rand_inds]
                else:
                    batch = np.zeros((inerf_batch_size, 2), dtype=np.int)
                    batch[:POI.shape[0]] = POI
                    rand_inds = np.random.choice(not_POI.shape[0], size=inerf_batch_size-POI.shape[0], replace=False)
                    batch[POI.shape[0]:] = not_POI[rand_inds]

            elif sampling_strategy == 'interest_regions':
                rand_inds = np.random.choice(interest_regions.shape[0], size=inerf_batch_size, replace=False)
                batch = interest_regions[rand_inds]

            else:
                print('Unknown sampling strategy')
                break

            
            target_s = obs_img_masked[batch[:, 1], batch[:, 0]]
            target_s = torch.Tensor(target_s).to(device)

            pose = cam_transf(start_pose)

            # to neuralangelo style
            pose[:3, 3] *= 1/scale # 位移缩小scale倍
            pose[:2] *= -1

            infer_data_inerf = infer_data.copy()
            
            batch = [i[0] + i[1] * obs_img_masked.shape[1] for i in batch]
            infer_data_inerf["ray_indices"] = np.array(batch)[None, ...] # check HW order 
            infer_data_inerf["pose"] = pose[None, :3, :]

            if trainer.cfg.trainer.ema_config.enabled:
                model = trainer.model.module.averaged_model
            else:
                model = trainer.model.module
            # ipdb.set_trace()
            model.eval()
            data_batches = []
            data = trainer.start_of_iteration(infer_data_inerf, current_iteration=trainer.current_iteration) 
            # data = to_cuda(data)
            
            # output = model.inference(data)

            # Render the full images.
            model.debugckpt = args.debugckpt
            output = model.render_image(data["pose"], data["intr"], image_size=model.cfg_data_wxz.test.image_size, # TODO: may cause error validation image
                                    stratified=False, sample_idx=data["idx"], ray_indices=data["ray_indices"], detach=False)  # [B,N,C]
            
            # Get full rendered RGB and depth images.
            # rot = data["pose"][..., :3, :3]  # [B,3,3]
            # normal_cam = -output["gradient"] @ rot.transpose(-1, -2)  # [B,HW,3]
            data.update(output)
            rgb = data["rgb"]
            # ipdb.set_trace()
            # all_return = trainer.inference_one_image(infer_data_inerf, reverse_xy=False)

            optimizer.zero_grad()
            loss = img2mse(rgb, target_s)
            # print("loss", loss.item())
            loss.backward()
            optimizer.step()

            new_lrate = lrate * (0.8 ** ((k + 1) / 100))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            if (k + 1) % 20 == 0 or k == 0:
                print('Step: ', k)
                print('Loss: ', loss)

                with torch.no_grad():
                    pose_dummy = pose.cpu().detach().numpy()
                    # langeolo to vrep style
                    pose_dummy[:3, 3] *= scale # 位移缩小scale倍
                    pose_dummy[:2] *= -1
                    # calculate angles and translation of the optimized pose
                    phi = np.arctan2(pose_dummy[1, 0], pose_dummy[0, 0]) * 180 / np.pi
                    theta = np.arctan2(-pose_dummy[2, 0], np.sqrt(pose_dummy[2, 1] ** 2 + pose_dummy[2, 2] ** 2)) * 180 / np.pi
                    psi = np.arctan2(pose_dummy[2, 1], pose_dummy[2, 2]) * 180 / np.pi
                    translation = np.sqrt(pose_dummy[0,3]**2 + pose_dummy[1,3]**2 + pose_dummy[2,3]**2)
                    print("current pose", pose_dummy)
                    #translation = pose_dummy[2, 3]
                    # calculate error between optimized and observed pose
                    # phi_error = abs(phi_ref - phi) if abs(phi_ref - phi)<300 else abs(abs(phi_ref - phi)-360)
                    # theta_error = abs(theta_ref - theta) if abs(theta_ref - theta)<300 else abs(abs(theta_ref - theta)-360)
                    # psi_error = abs(psi_ref - psi) if abs(psi_ref - psi)<300 else abs(abs(psi_ref - psi)-360)
                    # rot_error = phi_error + theta_error + psi_error
                    # translation_error = abs(translation_ref - translation)
                    # print('Rotation error: ', rot_error)
                    # print('Translation error: ', translation_error)
                    # print('-----------------------------------')

                if OVERLAY is True:
                    with torch.no_grad():
                        del infer_data_inerf["ray_indices"]
                        all_return = trainer.inference_one_image(infer_data_inerf, reverse_xy=False)
                        rgb = all_return["rgb_map"][0].cpu().numpy()
                        # C H W to H W C
                        rgb = np.transpose(rgb, (1, 2, 0))
                        # ipdb.set_trace()
                        rgb8 = to8b(rgb)
                        ref = to8b(obs_img_masked)
                        filename = os.path.join("inerf_output", str(k)+'.png')
                        dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                        imageio.imwrite(filename, dst)
                        imgs.append(dst)

        if OVERLAY is True:
            imageio.mimwrite(os.path.join("inerf_output", 'video.gif'), imgs, duration=6) #quality = 8 for mp4 format


    # Finalize training.
    trainer.finalize(cfg)

