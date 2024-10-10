from MODEL_pythonAPI import getandsetPose
import mask
import numpy as np
import os,cv2

# pose_coor_list=["pc0", "pc1","pc3"]
pose_coor_list=["pc0", "pc0","pc1","pc3"]
num_steps = [10,10,10]

cam_name = 'cammask'
obj_names = ["phone_clean_crop","phone_clean_crop0", "phone2p_crop", "fpc1_crop", "box1", "scene_real_crop"] # 最后一个是场景
scale=[0.14888766,0.14888766, 0.14836, 0.126, 0.4158*0.9, 1.8*0.8] # 虽然没用但是给 nerf 用的  是从vrep里拿到的

# BGR
palette = np.array([[0,50,0,60,255,60],[0,50,50,60,255,255],[50,0,50,255,20,255],[50,0,0,255,30,30],[0,0,50,30,30,255], [5,5,5,255,255,255]], dtype=np.uint8)
assert len(obj_names) == len(scale)
assert len(obj_names) == palette.shape[0]

for file in os.listdir('./outdir'):
    file_path = os.path.join('./outdir', file)
    if os.path.isfile(file_path):
        os.remove(file_path)

fn = getandsetPose.get_cam_image
fn_args = {"cam_name":cam_name, "obj_names":obj_names}
getandsetPose.vrep_timeloop(fn, fn_args, pose_coor_list=pose_coor_list, num_steps = num_steps,cam_name="cammask", set_obj_id=30)


# read timeid.txt use numpy
timeid = np.loadtxt(os.path.join("./outdir", f"timeid.txt")).astype(int)

for obj_id in range(len(obj_names)):
    for t in timeid:
        mask.mask_image(vrep_rgb_pth = f"./outdir/{t}{cam_name}.png", nerf_rgb_pth = f"./outdir/nerf_{cam_name}.png", lb=palette[obj_id, 0:3],ub=palette[obj_id, 3:], vrep_mask_outdir=f"./outdir/{t}{obj_names[obj_id]}_mask.png")
        # mask.mask_image(vrep_rgb_pth = f"./outdir/{t}{cam_name}.png", nerf_rgb_pth = f"./outdir/nerf_{cam_name}.png", lb=[5,0,0],ub=[255, 90, 90], vrep_mask_outdir=f"./outdir/{t}{obj_names[1]}_mask.png")

# merge mask
# import ipdb; ipdb.set_trace()
for t in timeid:
    base_img = None
    for i, obj_id in enumerate(obj_names):
            if i == len(obj_names)-1:
                continue # 最后一个是场景
            obj_mask = cv2.imread(f"./outdir/{t}{obj_id}_mask.png")
            if type(base_img) == type(None):
                base_img = np.zeros_like(obj_mask)
            base_img[obj_mask[...,0]>0]=(palette[i][3:])/2
    cv2.imwrite(f"./outdir/{t}merged_mask_{cam_name}.png", base_img)
    scene_mask = np.ones_like(base_img)*255
    # import ipdb; ipdb.set_trace()
    scene_mask[base_img[...,0]>0] = 0
    scene_mask[base_img[...,1]>0] = 0
    scene_mask[base_img[...,2]>0] = 0
    cv2.imwrite(f"./outdir/{t}{obj_names[-1]}_mask.png", scene_mask)
# clear ./outdir folder
