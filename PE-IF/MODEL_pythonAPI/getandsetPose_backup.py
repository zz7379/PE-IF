import time
import sys,os
import numpy as np
import cv2
sys.path.append('./MODEL_pythonAPI/VREP_remoteAPIs')
import sim
# ------
# https://blog.csdn.net/weixin_41754912/article/details/82353012
# ------
def vrep_timeloop(fn, fn_args):
    cam_name = 'cammask'
    obj_name="note11"
    outdir="./outdir"

    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim
    if clientID == -1:
        raise Exception('Failed connecting to remote API server')
    # 设置仿真步长，为了保持API端与V-rep端相同步长

    return_code, obj_handle = sim.simxGetObjectHandle(clientID, obj_name, sim.simx_opmode_blocking)
    return_code, visionsensor_handle = sim.simxGetObjectHandle(clientID, cam_name, sim.simx_opmode_blocking)
    print("obj_handle =",obj_handle,"visionsensor_handle =",visionsensor_handle)

    tstep = 0.02
    sim.simxSetFloatingParameter(clientID, sim.sim_floatparam_simulation_time_step, tstep, sim.simx_opmode_oneshot)
    # 然后打开同步模式
    sim.simxSynchronous(clientID, True)
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
    
    lastCmdTime=sim.simxGetLastCmdTime(clientID)  # 记录当前时间
    sim.simxSynchronousTrigger(clientID)  # 让仿真走一步
    # 开始仿真

    opmod = sim.simx_opmode_streaming

    _, obj_pos = sim.simxGetObjectPosition(clientID, obj_handle, -1, opmod)
    _, obj_ori = sim.simxGetObjectOrientation(clientID, obj_handle, -1, opmod)

    _, visionsensor_pos = sim.simxGetObjectPosition(clientID, visionsensor_handle, -1, opmod)
    _, visionsensor_ori = sim.simxGetObjectOrientation(clientID, visionsensor_handle, -1, opmod)

    _, relative_pos = sim.simxGetObjectPosition(clientID, obj_handle, visionsensor_handle, opmod)
    _, relative_ori = sim.simxGetObjectOrientation(clientID, obj_handle, visionsensor_handle, opmod)

    print(f"The position of {obj_name}: ", [i for i in obj_pos])
    print(f"The orientation of {obj_name}: ", [round(i / np.pi * 180, 4) for i in obj_ori])
    
    print(f"The position of vision sensor: ", [i for i in visionsensor_pos])
    print(f"The orientation of vision sensor: ", [round(i / np.pi * 180, 4) for i in visionsensor_ori])

    print(f"The position of obj2cam: ", [i for i in relative_pos])
    print(f"The orientation of obj2cam: ", [round(i / np.pi * 180, 4) for i in relative_ori])

    errprCode, resolution, image = sim.simxGetVisionSensorImage(clientID, visionsensor_handle, 0, opmod)


    while sim.simxGetConnectionId(clientID) != -1:

        currCmdTime=sim.simxGetLastCmdTime(clientID)  # 记录当前时间
        print("time =",currCmdTime)
        dt = currCmdTime - lastCmdTime # 记录时间间隔，用于控制
        #-----------------------------------

        opmod = sim.simx_opmode_buffer

        _, obj_pos = sim.simxGetObjectPosition(clientID, obj_handle, -1, opmod)
        _, obj_ori = sim.simxGetObjectOrientation(clientID, obj_handle, -1, opmod)

        _, visionsensor_pos = sim.simxGetObjectPosition(clientID, visionsensor_handle, -1, opmod)
        _, visionsensor_ori = sim.simxGetObjectOrientation(clientID, visionsensor_handle, -1, opmod)

        _, relative_pos = sim.simxGetObjectPosition(clientID, obj_handle, visionsensor_handle, opmod)
        _, relative_ori = sim.simxGetObjectOrientation(clientID, obj_handle, visionsensor_handle, opmod)

        print(f"The position of {obj_name}: ", [i for i in obj_pos])
        print(f"The orientation of {obj_name}: ", [round(i / np.pi * 180, 4) for i in obj_ori])
        
        print(f"The position of vision sensor: ", [i for i in visionsensor_pos])
        print(f"The orientation of vision sensor: ", [round(i / np.pi * 180, 4) for i in visionsensor_ori])

        print(f"The position of obj2cam: ", [i for i in relative_pos])
        print(f"The orientation of obj2cam: ", [round(i / np.pi * 180, 4) for i in relative_ori])

        if opmod == sim.simx_opmode_streaming or relative_ori[0] == 0:
            continue
        # write relative_pos and relative_ori to txt file
        with open(os.path.join(outdir, f"{obj_name}to{cam_name}.txt"), 'w') as file:
            for i in relative_pos:
                file.write(f"{i} ")
            for i in relative_ori:
                file.write(f"{i} ")
        errprCode, resolution, image = sim.simxGetVisionSensorImage(clientID, visionsensor_handle, 0, opmod)
        
        sensorFrame = np.array(image)
        sensorFrame = sensorFrame.astype(np.uint8)
        sensorFrame.resize([resolution[1], resolution[0], 3]) 
        sensorFrame = cv2.flip(sensorFrame, 0) 
        sensorFrame = sensorFrame[:, :, ::-1]
        
        if cv2.imwrite(os.path.join(outdir, f"{cam_name}.png"), sensorFrame):
            print("Image saved to ", os.path.join(outdir, f"{cam_name}.png"))
            print("Image shape: ", sensorFrame.shape)
        else:
            print("Failed to save image")

        #------------
        lastCmdTime=currCmdTime    # 记录当前时间
        sim.simxSynchronousTrigger(clientID)  # 进行下一步
        sim.simxGetPingTime(clientID) 
        # import ipdb; ipdb.set_trace()

if __name__ == "__main__":

    vrep_timeloop(None, None)
