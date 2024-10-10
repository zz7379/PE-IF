import sys, torch, cv2
sys.path.append('./MODEL_pythonAPI/VREP_remoteAPIs')
import time
import math, sim, ipdb
import numpy as np
from util import camera

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
''' Arm control function '''
def VREP_armControl(sim, clientID, arm_joints_handle, desired_arm_joint_angles):
    jointsNum = len(arm_joints_handle)
    for i in range(0,jointsNum):
        sim.simxSetJointPosition(clientID, arm_joints_handle[i], desired_arm_joint_angles[i], sim.simx_opmode_blocking)

def get_sensor_img(clientID, visionsensor_handle, pth):
    errprCode = 1
    while errprCode:
        errprCode, resolution, image = sim.simxGetVisionSensorImage(clientID, visionsensor_handle, 0,
                                                                        sim.simx_opmode_blocking)
        # print("get image errcode", errprCode, len(image))
        if errprCode == 1:
            time.sleep(0.3)
        # ipdb.set_trace()

    # clip image to [0, 255]
    sensorFrame = np.array(image)
    # sensorFrame += 128
    sensorFrame = sensorFrame.astype(np.uint8)
    # ipdb.set_trace()
    sensorFrame.resize([resolution[1], resolution[0], 3])  # 调整图像通道结构
    sensorFrame = cv2.flip(sensorFrame, 0)  # 翻转图像
    # cv2.imshow('', sensorFrame)

    # import ipdb; ipdb.set_trace()
    # turn RGB to BGR
    sensorFrame = sensorFrame[:, :, ::-1]
    cv2.imwrite(pth, sensorFrame)
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

def cam_interp(clientID, pose_coor_list=["pc0", "pc1"], num_steps = [10]):
    opmode = sim.simx_opmode_blocking
    vrep_xyzabg = []
    vrep_Ms = []
    interp_abg_radius = [] # radius
    pose_coor_id = []
    interp_trans = []
    interp_xyzabg = []
    for pose_name in pose_coor_list:
        err_code = 1
        while not err_code == 0:
            err_code, id = sim.simxGetObjectHandle(clientID, pose_name ,opmode)
        print("interp_cam_id:", err_code, id)
        pose_coor_id.append(id)

    for id in pose_coor_id:
        # import ipdb; ipdb.set_trace()
        err_code = 1
        while err_code:
            err_code, pose_pos = sim.simxGetObjectPosition(clientID, id, -1, opmode)
        err_code = 1
        while err_code:
            err_code, pose_ori = sim.simxGetObjectOrientation(clientID, id, -1, opmode)
        vrep_M = torch.eye(4) 
        vrep_M[:3, 3] = torch.tensor(pose_pos)
        vrep_M[:3, :3] = XYZ_angle_to_rotation_matrix(pose_ori)
        vrep_xyzabg.append(pose_pos+pose_ori)
        vrep_Ms.append(vrep_M)
    assert len(num_steps) == len(vrep_Ms) - 1
    P = camera.Pose()
    # interp matrix and get interp_cam_poses
    vrep_xyzabg = np.array(vrep_xyzabg)

    for i in range(len(num_steps)):
        for t in range(num_steps[i]):
            cam_input = P.interpolate(vrep_Ms[i][:3], vrep_Ms[i+1][:3], t/num_steps[i])
            abg = vrep_xyzabg[i][3:6] * (1-t/num_steps[i]) + vrep_xyzabg[i+1][3:6] * t/num_steps[i]
            interp_abg_radius.append(abg.tolist())
            interp_trans.append(cam_input[:3, 3])
            interp_xyzabg.append(cam_input[:3, 3].tolist()+abg.tolist())
            # cam_poses_render.append(cam_input[:3, :])
    # save interp_xyzabg to txt
    np.savetxt("./outdir/interp_xyzabg.txt",np.array(interp_xyzabg))
    return np.array(interp_xyzabg)
    

if __name__ == '__main__':

    ''' Initialization '''
    cam_name = 'cam3'
    obj_name = 'K50noscale'
    pose_coor_list = ["pc0", "pc1"]
    num_steps = [10]
    time_step = 0.1
    opmode = sim.simx_opmode_blocking

    print ('Program started')
    
    
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
    if clientID != -1:
        return_code, cam_id = sim.simxGetObjectHandle(clientID, cam_name,opmode)
        return_code, obj_id = sim.simxGetObjectHandle(clientID, obj_name,opmode)
        nerf_render_poses = []
        
        interp_abg_radius = [] # radius
        pose_coor_id = []
        interp_trans = []
        get_sensor_img(clientID, cam_id, f"./vrep_img/vrep_{0}.png")
        for pose_name in pose_coor_list:
            err_code = 1
            while not err_code == 0:
                err_code, id = sim.simxGetObjectHandle(clientID, pose_name ,opmode)
            # ipdb.set_trace()
            print("id:", err_code, id)
            pose_coor_id.append(id)

        # pose_coor_id =[193,195,197]
        # pose_id to matrix
        vrep_xyzabg = []
        vrep_Ms = []
        for id in pose_coor_id:
            # import ipdb; ipdb.set_trace()
            err_code = 1
            while err_code:
                err_code, pose_pos = sim.simxGetObjectPosition(clientID, id, -1, opmode)
            err_code = 1
            while err_code:
                err_code, pose_ori = sim.simxGetObjectOrientation(clientID, id, -1, opmode)
            vrep_M = torch.eye(4) 
            vrep_M[:3, 3] = torch.tensor(pose_pos)
            vrep_M[:3, :3] = XYZ_angle_to_rotation_matrix(pose_ori)
            vrep_xyzabg.append(pose_pos+pose_ori)
            vrep_Ms.append(vrep_M)

        assert len(num_steps) == len(vrep_Ms) - 1
        P = camera.Pose()
        # interp matrix and get interp_cam_poses
        vrep_xyzabg = np.array(vrep_xyzabg)

        for i in range(len(num_steps)):
            for t in range(num_steps[i]):
                cam_input = P.interpolate(vrep_Ms[i][:3], vrep_Ms[i+1][:3], t/num_steps[i])
                # interp_abg_radius.append(ZYX_matrix_to_euler(cam_input)) # radius
                abg = vrep_xyzabg[i][3:6] * (1-t/num_steps[i]) + vrep_xyzabg[i+1][3:6] * t/num_steps[i]
                interp_abg_radius.append(abg.tolist())
                interp_trans.append(cam_input[:3, 3])
                # cam_poses_render.append(cam_input[:3, :])
                print("----------interpolate_M----------")
                print(i, t, cam_input[:3, :])
        
        xyzabg = []
        

        # cam_poses_vrep = torch.stack(cam_poses_render, dim=0)
        

        # tensor_pose_cam_pose = torch.tensor(pose_cam_pose)
        simu_time = 0
        # while simu_time < 3:
        for i in range(len(interp_abg_radius)):
            # print(i, "T=", interp_trans[i], "abg=", interp_abg_radius[i])
            # xyzabg.append(interp_trans[i].numpy().tolist()+interp_abg_radius[i])
            time.sleep(0.5)
            _ = 1
            while _:
                _ = sim.simxSetObjectPosition(clientID, cam_id, -1, interp_trans[i], opmode)
            _ = 1
            while _:
                _ = sim.simxSetObjectOrientation(clientID, cam_id, -1, interp_abg_radius[i], opmode)
            _ = 1
            while _:
                _, obj2cam_pos = sim.simxGetObjectPosition(clientID, obj_id, cam_id,
                                                              opmode)
            _ = 1
            while _:
                _, obj2cam_ori = sim.simxGetObjectOrientation(clientID, obj_id, cam_id,
                                                                 opmode)
            xyzabg.append(obj2cam_pos + obj2cam_ori)
            get_sensor_img(clientID, cam_id, f"./vrep_img/vrep_{i}.png")

            obj2cam_matrix = torch.eye(4)
            obj2cam_matrix[:3, 3] = torch.tensor(obj2cam_pos).T
            obj2cam_matrix[:3, :3] = XYZ_angle_to_rotation_matrix(obj2cam_ori)
            # print(interp_trans[i])
            print("nerf render pose", obj2cam_matrix)
            nerf_render_poses.append(obj2cam_matrix)
            
            # _, visionsensor_pos = sim.simxGetObjectPosition(clientID, visionsensor_handle, -1,
            #                                                   sim.simx_opmode_streaming)
            # _, visionsensor_ori = sim.simxGetObjectOrientation(clientID, visionsensor_handle, -1,
            #                                                      sim.simx_opmode_streaming)

            simu_time = simu_time + 0.2

    else:
        print('Failed connecting to remote API server')
    np.savetxt("./xyzabg.txt",np.array(xyzabg))


    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
    print('Program ended')