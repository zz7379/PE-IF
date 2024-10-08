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


def inerf_inference(trainer, template_intr, prev_xyzabg_path, obs_img_path, scale, vrep_mask_path=None, sampling_strategy = "random", debugpose=None):

    xyzabg = torch.tensor(np.loadtxt(prev_xyzabg_path), dtype=torch.float32).reshape(-1, 6) * 1.05
    obs_img = cv2.imread(obs_img_path)
    vrep_mask = cv2.imread(vrep_mask_path)
    lrate = 0.01 
    inerf_batch_size = 512
    dil_iter = 3
    iters = 1000
    save_interval = 100


    obs_img = (np.array(obs_img) / 255.).astype(np.float32)
    if debugpose is not None:
        # import ipdb; ipdb.set_trace()
        pose = torch.eye(4)
        pose[:3, :] = debugpose
        # jitter pose
        pose[:3, 3] *= torch.randn(3) * 0.1 + 1
        pose[:3, :3] = pose[:3, :3] @ XYZ_angle_to_rotation_matrix(torch.randn(3) * 0.1)
        pose[:2] *= -1

    else:
        pose = torch.eye(4)
        pose[:3, :3] = XYZ_angle_to_rotation_matrix(xyzabg[0, 3:])
        pose[:3, 3] = xyzabg[0, :3]
    cam_poses_render = pose[None, :3, :]
    cam_intr_render = template_intr.clone()[None, ...]
    infer_data = {"pose": cam_poses_render, "intr": cam_intr_render, "idx": torch.arange(1), "file_prefix": f"{0}"}
    # trainer.inference_one_image(infer_data, reverse_xy=False)

    start_pose = pose.clone().detach()
    
    # downsample obs_img to mask size
    if type(vrep_mask) != type(None):
        # vrep_mask = cv2.resize(vrep_mask, (obs_img.shape[1], obs_img.shape[0]), interpolation=cv2.INTER_AREA)
        obs_img = cv2.resize(obs_img, (vrep_mask.shape[1], vrep_mask.shape[0]), interpolation=cv2.INTER_AREA)
        dilated_vrep_mask = cv2.dilate(vrep_mask, np.ones((20, 20), np.uint8), iterations=3)
        dilated_vrep_mask[dilated_vrep_mask > 0] = 1
        cv2.imwrite("./pose_file/dilated_vrep_mask.png", dilated_vrep_mask)
        obs_img_masked = obs_img * dilated_vrep_mask
        cv2.imwrite("./pose_file/obs_img_masked.png", obs_img_masked)
        # make sure enough pixels are left
        # inerf_batch_size = (inerf_batch_size / (dilated_vrep_mask > 0).mean()).astype(int)
        # 
    else:
        obs_img_masked = obs_img     
        vrep_mask = np.ones_like(obs_img_masked)

    
    # turn obs_img_masked to grayscale
    obs_img_masked = (np.array(obs_img_masked) * 255).astype(np.uint8)
    POI = find_POI(obs_img_masked, 0)  # xy pixel coordinates of points of interest (N x 2) using SIFT
    obs_img_masked = (np.array(obs_img_masked) / 255.).astype(np.float32)
    
    # save obs_img_masked
    cv2.imwrite("./debugobs_img_masked.png", obs_img_masked*255)
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

    imgs = []

    for k in range(iters):
        if sampling_strategy == 'random':
            # keep coords in mask
            coords = coords[vrep_mask[coords[:, 1], coords[:, 0], 0] > 0]
            rand_inds = np.random.choice(coords.shape[0], size=inerf_batch_size, replace=False)
            batch = coords[rand_inds]

        elif sampling_strategy == 'full':
            # generate xy grid of size (H/2, W/2)
            downsample_scale = 16
            batch  = np.asarray(np.stack(np.meshgrid(np.linspace(0, 533//downsample_scale - 1, 533//downsample_scale), np.linspace(0, 300//downsample_scale - 1, 300//downsample_scale)), -1))
            batch *= downsample_scale
            batch = batch.reshape(-1, 2).astype(int)
            batch_full = batch
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
        # discard xy out of mask
        if type(vrep_mask) != type(None):
            # import ipdb; ipdb.set_trace()
            batch = batch[vrep_mask[batch[:, 1], batch[:, 0], 0] > 0]

        batch_backup = batch

        target_s = obs_img_masked[batch[:, 1], batch[:, 0]] 
        target_s = torch.Tensor(target_s).to(device)[..., [2,1,0]]# cv2 BGR to nerf RGB

        pose = cam_transf(start_pose)

        # to neuralangelo style
        pose[:3, 3] *= 1/scale # 位移缩小scale倍
        pose[:2] *= -1

        infer_data_inerf = infer_data.copy()
        
        batch = [i[0] + i[1] * obs_img_masked.shape[1] for i in batch]
        infer_data_inerf["ray_indices"] = np.array(batch)[None, ...] # check HW order 
        # ipdb.set_trace()
        infer_data_inerf["pose"] = pose[None, :3, :]

        if trainer.cfg.trainer.ema_config.enabled:
            model = trainer.model.module.averaged_model
        else:
            model = trainer.model.module
        model.eval()
        data_batches = []
        data = trainer.start_of_iteration(infer_data_inerf, current_iteration=trainer.current_iteration)
        # data = to_cuda(data)
        
        # output = model.inference(data)

        # Render image   (why directly use render_image()? because we need to get gradient)
        # ipdb.set_trace()
        output = model.render_image(data["pose"], data["intr"], image_size=model.cfg_data_wxz.test.image_size, # TODO: may cause error validation image
                                stratified=False, sample_idx=data["idx"], ray_indices=data["ray_indices"], detach=False)  # [B,N,C]
        # Get full rendered RGB and depth images.
        # rot = data["pose"][..., :3, :3]  # [B,3,3]
        # normal_cam = -output["gradient"] @ rot.transpose(-1, -2)  # [B,HW,3]

        # 假设ray_indices是按照HW顺序排列的(逐行扫描)
        data.update(output)
        rgb = data["rgb"]
        if sampling_strategy == 'full':
            tt = rgb[0].cpu().detach().numpy().reshape(300//downsample_scale, 533//downsample_scale, 3)
            cv2.imwrite("./inerf_output/rgb_rendered.png", tt[..., ::-1]*255)
            tt = target_s.reshape(300//downsample_scale, 533//downsample_scale, 3).cpu().numpy()
            cv2.imwrite("./inerf_output/target.png", tt[..., ::-1]*255)
        # import ipdb; ipdb.set_trace()
        

        # save rgb to file use opencv

        optimizer.zero_grad()
        loss = img2mse(rgb, target_s)
        loss.backward()
        optimizer.step()

        new_lrate = lrate * (0.8 ** ((k + 1) / 100))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if (k + 1) % save_interval == 0 or k == 0:
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
                phi_ref = np.arctan2(debugpose[1, 0], debugpose[0, 0]) * 180 / np.pi
                theta_ref = np.arctan2(-debugpose[2, 0], np.sqrt(debugpose[2, 1] ** 2 + debugpose[2, 2] ** 2)) * 180 / np.pi
                psi_ref = np.arctan2(debugpose[2, 1], debugpose[2, 2]) * 180 / np.pi
                translation_ref = np.sqrt(debugpose[0,3]**2 + debugpose[1,3]**2 + debugpose[2,3]**2)

                phi_error = abs(phi_ref - phi) if abs(phi_ref - phi)<300 else abs(abs(phi_ref - phi)-360)
                theta_error = abs(theta_ref - theta) if abs(theta_ref - theta)<300 else abs(abs(theta_ref - theta)-360)
                psi_error = abs(psi_ref - psi) if abs(psi_ref - psi)<300 else abs(abs(psi_ref - psi)-360)
                rot_error = phi_error + theta_error + psi_error
                translation_error = abs(translation_ref - translation)
                print('Rotation error: ', rot_error)
                print('Translation error: ', translation_error)
                print('-----------------------------------')


            with torch.no_grad():
                del infer_data_inerf["ray_indices"]
                all_return = trainer.inference_one_image(infer_data_inerf, reverse_xy=False)
                rgb = all_return["rgb_map"][0].cpu().numpy()
                # C H W to H W C
                rgb = np.transpose(rgb, (1, 2, 0))
                rgb8 = to8b(rgb)
                ref = to8b(obs_img_masked[..., [2,1,0]])
                filename = os.path.join("inerf_output", str(k)+'.png')
                dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)

                # see point of interest
                # dst[batch_backup[:,1],batch_backup[:,0],:] = [0, 0, 255]
                print("writting to", filename)
                imageio.imwrite(filename, dst)
                imgs.append(dst)


    imageio.mimwrite(os.path.join("inerf_output", 'video.gif'), imgs, duration=6) #quality = 8 for mp4 format
    return pose_dummy

def render_by_xyzabg(trainer, H=300, W=533, xyzabg_path="pose_file/xyzabg.txt", vrep_mask=None, scale=0.114265990, file_prefix=None):


    trainer.current_epoch = trainer.checkpointer.resume_epoch or trainer.current_epoch
    trainer.current_iteration = trainer.checkpointer.resume_iteration or trainer.current_iteration
    trainer.mode = 'val'
    template_data = torch.load("eval_data.pt")
    template_intr = template_data["intr"][0]


    template_intr[0, 2] *= W / template_data['image'].shape[3]
    template_intr[1, 2] *= H / template_data['image'].shape[2]

    print("intr is ", template_intr)
    FOV_W = 2 * math.atan(0.5 * W / template_intr[0, 0])
    FOV_H = 2 * math.atan(0.5 * H / template_intr[1, 1])
    print("FOV_W, FOV_H", FOV_W * 180 / np.pi, FOV_H * 180 / np.pi)

    # ----------- get interped xyzab and render---------------

    cam_poses_render = []
    cam_intr_render = []

    render_xyzabg = torch.tensor(np.loadtxt(xyzabg_path), dtype=torch.float32).reshape(-1, 6)
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
        if not type(vrep_mask) == type(None):
            assert cam_poses_vrep.shape[0] == 1
            ray_indices = np.where(vrep_mask[...,0] > 0)
            ray_indices = ray_indices[1] + ray_indices[0] * vrep_mask.shape[1]
            ray_indices = np.array(ray_indices)[None, ...]
            # ipdb.set_trace()
        else:
            ray_indices = None
        infer_data = {"pose": cam_poses_vrep[i:(i+1)], "intr": cam_intr_vrep[i:(i+1)], "idx": torch.arange(1), "file_prefix": f"{file_prefix}{i}"}
        with torch.no_grad():
            trainer.inference_one_image(infer_data, reverse_xy=False, ray_indices=ray_indices)
    
def get_trainer_and_intr(args, template_data_path):
    cfg_cmd = []
    cfg = Config(args.config)
    cfg.data.val.image_size = cfg.data.test.image_size
    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)
    cfg.logdir = init_logging(args.config, args.logdir, makedir=True)

    # print(cfg)
    trainer = get_trainer(cfg, is_inference=args.inference or args.test, seed=args.seed) # FIXME: not sure if  is is_inference=args.inference
    trainer.set_data_loader(cfg, split="train")
    trainer.set_data_loader(cfg, split="val")
    trainer.checkpointer.load(args.checkpoint, args.resume, load_sch=True, load_opt=True)
    trainer.mode = 'train'

    
    # W, H = (533, 300)
    # W, H = (300, 300)
    H, W = cfg.data.test.image_size # check check
    trainer.current_epoch = trainer.checkpointer.resume_epoch or trainer.current_epoch
    trainer.current_iteration = trainer.checkpointer.resume_iteration or trainer.current_iteration
    trainer.mode = 'val'
    template_data = torch.load(template_data_path)
    template_intr = template_data["intr"][0]

    template_pose = template_data["pose"][0]
    template_intr[0, 2] *= W / template_data['image'].shape[3]
    template_intr[1, 2] *= H / template_data['image'].shape[2]

    FOV_W = 2 * math.atan(0.5 * W / template_intr[0, 0])
    FOV_H = 2 * math.atan(0.5 * H / template_intr[1, 1])
    # print("FOV_W, FOV_H", FOV_W * 180 / np.pi, FOV_H * 180 / np.pi)
    return trainer, template_intr, W, H

if __name__=='__main__':
    # -----vrep + nerf--------
    # prepare vrepoutput by run main.py in thinkbook
    # vrepoutput contains timeid.txt, pose_file, vrep_mask
    # ATTENTION: check the intrincis of vrep camera is identical to nerf camera
    # ATTNETION: when training nerf, keep debugckpt on! Because sometimes cpkt fails to save. 
    # ATTENTION: when export mesh, keep noscale on!
    # ATTENTION: when import mesh into vrep, keep no scale and up-vector Z.

    INERF_ON = False # if True, only run inerf

    clean_folder("outdir")

    cam_name = 'cammask'
    obj_names = ["phone_clean_crop","phone_clean_crop0", "fixer", "fpc1_crop", "box1", "scene_real_crop"] # 最后一个是场景
    scale=[0.14888766,0.14888766, 0.12960, 0.126, 0.4158, 1.8] # nerf2vrep scale

    obj_names = ["phone_clean_crop","phone_clean_crop0", "phone2p_crop", "fpc1_crop", "box1", "scene_real_crop"] # 最后一个是场景
    scale=[0.14888766,0.14888766, 0.14836, 0.126, 0.4158*0.9, 1.8*0.8] 
    
    # BGR   palette for vrep mask
    palette = np.array([[0,50,0,60,255,60],[0,50,50,60,255,255],[50,0,50,255,20,255],[50,0,0,255,30,30],[0,0,50,30,30,255], [5,5,5,255,255,255]], dtype=np.uint8)

    timeid = np.loadtxt("vrepoutput/timeid.txt", dtype=np.int32)

    # obj_names = [obj_names[-1]]
    # scale=[scale[-1]]
    # palette = [palette[-1]]
    # timeid = [timeid[0],timeid[-1]]


    args, cfg_cmd = parse_args()
    assert cfg_cmd == [], "Unknown arguments: {}".format(cfg_cmd)
    set_affinity(args.local_rank)
    set_random_seed(args.seed, by_rank=True)
    imaginaire.config.DEBUG = args.debug
    cfg = Config(args.config)
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)
    for i, obj_name in enumerate(obj_names):

        # print("========obj_name", obj_name)
        if obj_name in ["phone_clean_crop", "phone_clean_crop0"]:
            class_name = "phone_clean"
            args.checkpoint = "logs/phone_clean/epoch_05263_iteration_000100000_checkpoint.pt"
        elif obj_name in ["fixer"]:
            class_name = "fixer"
            args.checkpoint = "logs/fixer/epoch_18000_iteration_000090000_checkpoint.pt"
        elif obj_name in ["fpc1_crop"]:
            class_name = "fpc1"
            args.checkpoint = "logs/fpc1/epoch_05555_iteration_000100000_checkpoint.pt"
        elif obj_name in ["box1"]:
            class_name = "box1"
            args.checkpoint = "logs/box1/epoch_11500_iteration_000230000_checkpoint.pt"
        elif obj_name in ["scene_real_crop"]:
            class_name = "scene_real"
            args.checkpoint = "logs/scene_real/epoch_02112_iteration_000150000_checkpoint.pt"
            # args.checkpoint = "logs/scene_real2/epoch_50000_iteration_000100000_checkpoint.pt"
        elif obj_name in ["phone2p_crop"]:
            class_name = "phone2p"
            args.checkpoint = "logs/phone2p/epoch_07142_iteration_000100000_checkpoint.pt"
        else: 
            raise NotImplementedError
        # with open(f'logs/{class_name}/latest_checkpoint.txt', 'r') as f:
        #     text = f.read()
        # args.checkpoint=f'./logs/{class_name}/' + text
        print(class_name)
        if INERF_ON and not class_name in ["fixer"]:
            continue
        args.config=f'./logs/{class_name}/config.yaml'
        args.logdir=f'logs/{class_name}'

        # import ipdb; ipdb.set_trace()
        # trainer, template_intr, W, H = get_trainer_and_intr(args, template_data_path=f"intr_{class_name}.pt")
        trainer, template_intr, W, H = get_trainer_and_intr(args, template_data_path=f"intr_phone_clean.pt")
        # --------------- nerf ready ,inerf ----------------

        # real inerf
        # inerf_pose = inerf_inference(trainer, template_intr, prev_xyzabg_path=f"pose_file/inerf_xyzabg.txt", obs_img_path="pose_file/Image1.png", vrep_mask_path= "pose_file/vrep_mask.png", sampling_strategy = "random", scale=scale)
        
        # # fake inerf
        if INERF_ON:

            image_temp = torch.load(f"pose_file/{class_name}_image_temp.pth")[0]
            intr_temp = torch.load("pose_file/intr_temp.pth")[0]
            pose_temp = torch.load(f"pose_file/{class_name}_pose_temp.pth")[0]
            # save image_temp into png
            image_temp = image_temp.cpu().numpy()
            image_temp = np.transpose(image_temp, (1, 2, 0))
            image_temp = to8b(image_temp)
            cv2.imwrite("image_temp.png", image_temp[..., ::-1])
            inerf_pose = inerf_inference(trainer, intr_temp, debugpose=pose_temp, prev_xyzabg_path=f"pose_file/inerf_xyzabg.txt", obs_img_path="image_temp.png", sampling_strategy = "random", scale=1)
            # render single image
            # render_by_xyzabg(trainer, H=300, W=533, xyzabg_path="pose_file/inerf_xyzabg.txt", vrep_mask=None, scale=scale)
            continue

        # --------------------------------------------

        # read txt into numpy

        for time in timeid:
            # vrep_mask = cv2.imread(f"vrepoutput/{time}{obj_name}_mask.png") # TODO: fix this
            vrep_mask = None
            render_by_xyzabg(trainer, H=H, W=W, xyzabg_path=f"vrepoutput/{time}{obj_name}to{cam_name}.txt", vrep_mask=vrep_mask, scale=scale[i], file_prefix=f"{time}{obj_name}")

    for modal in ["image", "normal"]:
        imgs = []
        for time in timeid:
            vrep_rgb = cv2.imread(f"vrepoutput/{time}{cam_name}.png")
            for obj_name in obj_names:
                vrep_mask = cv2.imread(f"vrepoutput/{time}{obj_name}_mask.png")

                vrep_mask = cv2.dilate(vrep_mask, np.ones((2, 2), np.uint8), iterations=2)
                nerf_rgb = cv2.imread(f"outdir/{time}{obj_name}0_{modal}1.png")
                if obj_name == "scene_real_crop" and modal == "image":
                    # change the brightness of nerf_rgb
                    nerf_rgb = nerf_rgb.astype(np.float32)
                    nerf_rgb *= 2
                    nerf_rgb[nerf_rgb > 255] = 255
                    nerf_rgb = nerf_rgb.astype(np.uint8)


                vrep_rgb[np.where(vrep_mask == 255)] = nerf_rgb[np.where(vrep_mask == 255)]
            cv2.imwrite(f"outdir/0000_{time}{cam_name}_{modal}_mixed.png", vrep_rgb)
            imgs.append(vrep_rgb[...,[2,1,0]])

        # import ipdb; ipdb.set_trace()
        imageio.mimwrite(os.path.join("outdir", f'{modal}_video.gif'), imgs, duration=5) #quality = 8 for mp4 format
        try:
            imageio.mimwrite(os.path.join("outdir", f'{modal}_video.mp4'), imgs, duration=5, quality = 8)
        except:
            pass




