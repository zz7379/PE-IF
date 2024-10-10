import time
import sys,os
import numpy as np
import cv2
sys.path.append('./MODEL_pythonAPI/VREP_remoteAPIs')

def get_cam_image(cam_name = 'cam3', obj_name="note11", outdir="./outdir"):

    import sim
    # from VREP_remoteAPIs import sim

    ''' Initialization '''
    print('Program started')
    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim


    if clientID != -1:
        print('Connected to remote API server')

        # get the pose of note11
        return_code, obj_handle = sim.simxGetObjectHandle(clientID, obj_name, sim.simx_opmode_blocking)
        return_code, visionsensor_handle = sim.simxGetObjectHandle(clientID, cam_name, sim.simx_opmode_blocking)

        _, obj_pos = sim.simxGetObjectPosition(clientID, obj_handle, -1, sim.simx_opmode_blocking)
        _, obj_ori = sim.simxGetObjectOrientation(clientID, obj_handle, -1, sim.simx_opmode_blocking)

        _, visionsensor_pos = sim.simxGetObjectPosition(clientID, visionsensor_handle, -1, sim.simx_opmode_blocking)
        _, visionsensor_ori = sim.simxGetObjectOrientation(clientID, visionsensor_handle, -1, sim.simx_opmode_blocking)

        _, relative_pos = sim.simxGetObjectPosition(clientID, obj_handle, visionsensor_handle, sim.simx_opmode_blocking)
        _, relative_ori = sim.simxGetObjectOrientation(clientID, obj_handle, visionsensor_handle, sim.simx_opmode_blocking)

        print(f"The position of {obj_name}: ", [i for i in obj_pos])
        print(f"The orientation of {obj_name}: ", [round(i / np.pi * 180, 4) for i in obj_ori])
        
        print(f"The position of vision sensor: ", [i for i in visionsensor_pos])
        print(f"The orientation of vision sensor: ", [round(i / np.pi * 180, 4) for i in visionsensor_ori])

        print(f"The position of obj2cam: ", [i for i in relative_pos])
        print(f"The orientation of obj2cam: ", [round(i / np.pi * 180, 4) for i in relative_ori])

        # write relative_pos and relative_ori to txt file
        with open(os.path.join(outdir, f"{obj_name}to{cam_name}.txt"), 'w') as file:
            for i in relative_pos:
                file.write(f"{i} ")
            for i in relative_ori:
                file.write(f"{i} ")
        errprCode, resolution, image = sim.simxGetVisionSensorImage(clientID, visionsensor_handle, 0, sim.simx_opmode_blocking)

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
        assert errprCode == 0

    else:
        print('Failed connecting to remote API server')

if __name__ == "__main__":
    get_cam_image()

    