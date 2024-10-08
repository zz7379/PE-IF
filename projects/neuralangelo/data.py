'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import json
import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
from PIL import Image, ImageFile

from projects.nerf.datasets import base
from projects.nerf.utils import camera
from projects.neuralangelo.utils.misc import gl_to_cv

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(base.Dataset):

    def __init__(self, cfg, is_inference=False, is_test=False):
        super().__init__(cfg, is_inference=is_inference, is_test=False)
        # is_test = False
        self.is_test = is_test
        self.cfg = cfg
        cfg_data = cfg.data
        self.root = cfg_data.root
        self.preload = cfg_data.preload
        self.H, self.W = cfg_data.val.image_size if is_inference else cfg_data.train.image_size

        if is_test:
            # self.gt_img_H, self.gt_img_W = (self.H, self.W)
            self.H, self.W = cfg_data.test.image_size
            self.test_poses = gen_rot_view(cfg.data.test.elev, cfg.data.test.max_angle, cfg.data.test.distance, cfg.data.test.size, cfg.data.test.y_axis)


        meta_fname = f"{cfg_data.root}/transforms.json"
        with open(meta_fname) as file:
            self.meta = json.load(file)
        self.list = self.meta["frames"]
        if cfg_data[self.split].subset:
            subset = cfg_data[self.split].subset
            subset_idx = np.linspace(0, len(self.list), subset+1)[:-1].astype(int)
            self.list = [self.list[i] for i in subset_idx]
        self.num_rays = cfg.model.render.rand_rays
        self.readjust = getattr(cfg_data, "readjust", None)
        # Preload dataset if possible.
        if cfg_data.preload and not is_test:
            self.images = self.preload_threading(self.get_image, cfg_data.num_workers)
            self.cameras = self.preload_threading(self.get_camera, cfg_data.num_workers, data_str="cameras")
        # import ipdb; ipdb.set_trace()
    def __len__(self):
        # import ipdb; ipdb.set_trace()
        return self.cfg.data.test.size if self.is_test else len(self.list)

    def __getitem__(self, idx):
        """Process raw data and return processed data in a dictionary.

        Args:
            idx: The index of the sample of the dataset.
        Returns: A dictionary containing the data.
                 idx (scalar): The index of the sample of the dataset.
                 image (R tensor): Image idx for per-image embedding.
                 image (Rx3 tensor): Image with pixel values in [0,1] for supervision.
                 intr (3x3 tensor): The camera intrinsics of `image`.
                 pose (3x4 tensor): The camera extrinsics [R,t] of `image`.
        """
        # Keep track of sample index for convenience.
        # import ipdb; ipdb.set_trace()
        sample = dict(idx=idx)
        if self.is_test:
            dummy_idx = 0
            dummy_image, dummy_image_size_raw = self.get_image(dummy_idx)
            dummy_image = self.preprocess_image(dummy_image, )

            intr, dummy_pose = self.get_camera(dummy_idx)
            pose = self.test_poses[idx]

            intr[0] *= (self.W / dummy_image_size_raw[0])
            intr[1] *= (self.H / dummy_image_size_raw[1])

            pose = torch.tensor(pose[0:3, :], device=dummy_pose.device, dtype=dummy_pose.dtype)
            print(pose)
            sample.update(
                image=dummy_image,
                intr=intr,
                pose=pose,
            )

            # import ipdb; ipdb.set_trace()
            # print(pose.shape, "\n", dummy_pose.shape)

        else:
            # Get the images.
            image, image_size_raw = self.images[idx] if self.preload else self.get_image(idx)
            image = self.preprocess_image(image)
            # Get the cameras (intrinsics and pose).
            # import ipdb; ipdb.set_trace()

            intr, pose = self.cameras[idx] if self.preload else self.get_camera(idx)
            intr, pose = self.preprocess_camera(intr, pose, image_size_raw)
            # print(self.H, self.W)
            # print("pose",pose,"\nintr",intr)
            
            # Pre-sample ray indices.
            if self.split == "train":
                ray_idx = torch.randperm(self.H * self.W)[:self.num_rays]  # [R]
                image_sampled = image.flatten(1, 2)[:, ray_idx].t()  # [R,3]
                sample.update(
                    ray_idx=ray_idx,
                    image_sampled=image_sampled,
                    intr=intr,
                    pose=pose,
                )
            else:  # keep image during inference
                sample.update(
                    image=image,
                    intr=intr,
                    pose=pose,
                )
        return sample

    def get_image(self, idx):
        fpath = self.list[idx]["file_path"]
        image_fname = f"{self.root}/{fpath}"
        image = Image.open(image_fname)
        image.load()
        image_size_raw = image.size
        return image, image_size_raw

    def preprocess_image(self, image):
        # Resize the image.
        image = image.resize((self.W, self.H))
        image = torchvision_F.to_tensor(image)
        rgb = image[:3]
        return rgb

    def get_camera(self, idx):
        # Camera intrinsics.
        intr = torch.tensor([[self.meta["fl_x"], self.meta["sk_x"], self.meta["cx"]],
                             [self.meta["sk_y"], self.meta["fl_y"], self.meta["cy"]],
                             [0, 0, 1]]).float()
        # Camera pose.
        c2w_gl = torch.tensor(self.list[idx]["transform_matrix"], dtype=torch.float32)
        c2w = gl_to_cv(c2w_gl)
        if not self.meta['centered']:
            # center scene
            center = np.array(self.meta["sphere_center"])
            if self.readjust:
                center += np.array(getattr(self.readjust, "center", [0]))
            c2w[:3, -1] -= center
        if not self.meta['scaled']:
            # scale scene
            scale = np.array(self.meta["sphere_radius"])
            if self.readjust:
                scale *= getattr(self.readjust, "scale", 1.)
            c2w[:3, -1] /= scale
        w2c = camera.Pose().invert(c2w[:3])
        return intr, w2c

    def preprocess_camera(self, intr, pose, image_size_raw):
        # Adjust the intrinsics according to the resized image.
        intr = intr.clone()
        raw_W, raw_H = image_size_raw
        intr[0] *= self.W / raw_W
        intr[1] *= self.H / raw_H
        return intr, pose


def gen_rot_view(elev, max_angle, distance, size, axis=[0, 1, 0]):

    # bg_to_cam

    cam_traj = get_rotating_cam(size, distance=distance, max_angle=max_angle)
    cam_elev = get_object_to_camera_matrix(elev, axis, 0)[None]
    cam_traj = cam_traj @ cam_elev
    # # field2cam = create_field2cam(cam_traj, field2cam_fr.keys())

    # camera_int = np.zeros((len(frameid_sub), 4))

    # # focal length = img height * distance / obj height
    # camera_int[:, :2] = opts["render_res"] * 2 * 0.8  # zoom out a bit
    # camera_int[:, 2:] = opts["render_res"] / 2
    return cam_traj

def get_rotating_cam(num_cameras, axis=[0, 1, 0], distance=3, initial_angle=0, max_angle=360):
    """Generate camera sequence rotating around a fixed object

    Args:
        num_cameras (int): Number of cameras in sequence
        axis (ndarray): (3,) Axis of rotation
        distance (float): Distance from camera to object
        initial_angle (float): Initial rotation angle, degrees (default 0)
        max_angle (float): Final rotation angle, degrees (default 360)
    Returns:
        extrinsics (ndarray): (num_cameras, 3, 4) Sequence of camera extrinsics
    """
    angles = np.linspace(initial_angle, max_angle, num_cameras)
    extrinsics = np.zeros((num_cameras, 4, 4))
    for i in range(num_cameras):
        extrinsics[i] = get_object_to_camera_matrix(angles[i], axis, distance)
    return extrinsics
import cv2
def get_object_to_camera_matrix(theta, axis, distance):
    """Generate 3x4 object-to-camera matrix that rotates the object around
    the given axis

    Args:
        theta (float): Angle of rotation in radians.
        axis (ndarray): (3,) Axis of rotation
        distance (float): Distance from camera to object
    Returns:
        extrinsics (ndarray): (3, 4) Object-to-camera matrix
    """
    theta = theta / 180 * np.pi
    rt4x4 = np.eye(4)
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    # import ipdb; ipdb.set_trace()
    R, _ = cv2.Rodrigues(theta * axis)
    # t = np.asarray([0, 0, distance])
    t = distance * axis
    rtmat = np.concatenate((R, t.reshape(3, 1)), axis=1)
    rt4x4[:3, :4] = rtmat
    return rt4x4