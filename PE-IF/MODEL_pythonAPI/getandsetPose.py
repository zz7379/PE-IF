import time
import sys,os
import numpy as np
import cv2
sys.path.append('./MODEL_pythonAPI/VREP_remoteAPIs')
import sim
from cam_fly import cam_interp
# ------
# https://blog.csdn.net/weixin_41754912/article/details/82353012
# ------
def vrep_timeloop(fn, fn_args,pose_coor_list=["pc1", "pc0"], num_steps = [30], cam_name="cammask",set_obj_id=19):

    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim
    if clientID == -1:
        raise Exception('Failed connecting to remote API server')
    # 设置仿真步长，为了保持API端与V-rep端相同步长,注意保持vrep开始仿真
    tstep = 0.001
    sim.simxSetFloatingParameter(clientID, sim.sim_floatparam_simulation_time_step, tstep, sim.simx_opmode_oneshot)
    # 然后打开同步模式
    sim.simxSynchronous(clientID, True)

    # sim.simxStopSimulation(clientID,sim.simx_opmode_blocking)
    # sim.simxLoadScene(clientID,scene_path,1,sim.simx_opmode_blocking)
    # import ipdb; ipdb.set_trace()
    
    # -----开始采集mask之前的剧本-----------
    set_obj_id = set_obj_id # !!!!!!!!!!注意检查这个
    # sim.simxSetObjectPosition(clientID, set_obj_id, -1, [-2.7,0.295,1.3], sim.simx_opmode_blocking)
    # sim.simxSetObjectOrientation(clientID, set_obj_id, -1, [3.1415,0,-2], sim.simx_opmode_blocking)
    # sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
    # -------------------------------------
    
    # sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
    
    lastCmdTime=sim.simxGetLastCmdTime(clientID)  # 记录当前时间
    sim.simxSynchronousTrigger(clientID)  # 让仿真走一步
    # 开始仿真
    # _, relative_pos = sim.simxGetObjectPosition(clientID, 153, -1, sim.simx_opmode_streaming)
    fn(clientID, opmod=sim.simx_opmode_streaming, **fn_args)
    timeid = []
    # =========== interp pose ====================
    xyzabg_interp = cam_interp(clientID, pose_coor_list=pose_coor_list, num_steps = num_steps)
    return_code, cam_handle = sim.simxGetObjectHandle(clientID, cam_name, sim.simx_opmode_blocking)



    sim.simxSetObjectPosition(clientID, set_obj_id, -1, [-2.75,0.315,1.3], sim.simx_opmode_blocking)#-2.7,0.295,1.2
    sim.simxSetObjectOrientation(clientID, set_obj_id, -1, [3.1415,0,-2], sim.simx_opmode_blocking)
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
    while sim.simxGetConnectionId(clientID) != -1 and len(timeid) < np.sum(num_steps):

        # =========== interp pose ====================
        print(sim.simxGetObjectOrientation(clientID, cam_handle, -1, sim.simx_opmode_blocking))
        _ = sim.simxSetObjectPosition(clientID, cam_handle, -1, xyzabg_interp[len(timeid), 0:3], sim.simx_opmode_blocking)
        _ = sim.simxSetObjectOrientation(clientID, cam_handle, -1, xyzabg_interp[len(timeid), 3:], sim.simx_opmode_blocking)
        print(sim.simxGetObjectOrientation(clientID, cam_handle, -1, sim.simx_opmode_blocking))
        # import ipdb; ipdb.set_trace()
        currCmdTime=sim.simxGetLastCmdTime(clientID)  # 记录当前时间
        timeid.append(currCmdTime)
        print("time =",currCmdTime)
        dt = currCmdTime - lastCmdTime # 记录时间间隔，用于控制
        
        fn(clientID, currCmdTime, **fn_args)

        lastCmdTime=currCmdTime    # 记录当前时间
        sim.simxSynchronousTrigger(clientID)  # 进行下一步
        sim.simxGetPingTime(clientID)
    # import ipdb; ipdb.set_trace()
    with open(os.path.join("./outdir", f"timeid.txt"), 'w') as file:
        for i in timeid:
            file.write(f"{i} ")
def get_cam_image(clientID, currCmdTime=None, cam_name = 'cam3', obj_names=["note11"], outdir="./outdir", opmod = sim.simx_opmode_buffer):

    assert clientID != -1
    if type(obj_names) == str:
        obj_names = [obj_names]

    # get the pose of note11
    # opmod = sim.simx_opmode_streaming
    # opmod = sim.simx_opmode_blocking


    return_code, cam_handle = sim.simxGetObjectHandle(clientID, cam_name, sim.simx_opmode_blocking)
    _, cam_pos = sim.simxGetObjectPosition(clientID, cam_handle, -1, opmod)
    _, vcam_ori = sim.simxGetObjectOrientation(clientID, cam_handle, -1, opmod)
    print(f"The position of {cam_name}: ", [i for i in cam_pos])
    print(f"The orientation of {cam_name}: ", [round(i / np.pi * 180, 4) for i in vcam_ori])
    for obj_name in obj_names:
        return_code, obj_handle = sim.simxGetObjectHandle(clientID, obj_name, sim.simx_opmode_blocking)
        _, obj_pos = sim.simxGetObjectPosition(clientID, obj_handle, -1, opmod)
        _, obj_ori = sim.simxGetObjectOrientation(clientID, obj_handle, -1, opmod)
        _, relative_pos = sim.simxGetObjectPosition(clientID, obj_handle, cam_handle, opmod)
        _, relative_ori = sim.simxGetObjectOrientation(clientID, obj_handle, cam_handle, opmod)

        print(f"The position of {obj_name}: ", [i for i in obj_pos])
        print(f"The orientation of {obj_name}: ", [round(i / np.pi * 180, 4) for i in obj_ori])
        
        print(f"The position of obj2cam: ", [i for i in relative_pos])
        print(f"The orientation of obj2cam: ", [round(i / np.pi * 180, 4) for i in relative_ori])
        # import ipdb; ipdb.set_trace()
        with open(os.path.join(outdir, f"{currCmdTime}{obj_name}to{cam_name}.txt"), 'w') as file:
            for i in relative_pos:
                file.write(f"{i} ")
            for i in relative_ori:
                file.write(f"{i} ")

    errprCode, resolution, image = sim.simxGetVisionSensorImage(clientID, cam_handle, 0, opmod)
    if opmod == sim.simx_opmode_streaming or obj_pos[0]+ obj_ori[0]== 0:
        return
    # write relative_pos and relative_ori to txt file
    
    sensorFrame = np.array(image)
    sensorFrame = sensorFrame.astype(np.uint8)
    sensorFrame.resize([resolution[1], resolution[0], 3])
    sensorFrame = cv2.flip(sensorFrame, 0) 
    sensorFrame = sensorFrame[:, :, ::-1]
    
    if cv2.imwrite(os.path.join(outdir, f"{currCmdTime}{cam_name}.png"), sensorFrame):
        print("Image saved to ", os.path.join(outdir, f"{currCmdTime}{cam_name}.png"))
        # print("Image shape: ", sensorFrame.shape)
    else:
        print("Failed to save image")
    assert errprCode == 0


