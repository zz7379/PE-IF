import time
import sys
import numpy as np
import cv2
sys.path.append('./VREP_remoteAPIs')

def getObjectPosition(vrep_sim, clientID, ObjName):
    return_code, object_handle = vrep_sim.simxGetObjectHandle(clientID, ObjName, vrep_sim.simx_opmode_oneshot_wait)
    if (return_code == vrep_sim.simx_return_ok):
        print('get object handle ok.')

        _, obj_pos = vrep_sim.simxGetObjectPosition(clientID, object_handle, -1, vrep_sim.simx_opmode_oneshot_wait)
        _, obj_ori = vrep_sim.simxGetObjectOrientation(clientID, object_handle, -1, vrep_sim.simx_opmode_oneshot_wait)
        return obj_pos, obj_ori


def getVisionSensorRGBData(vrep_sim, clientID, visionSensorName):
    return_code, camera_handle = vrep_sim.simxGetObjectHandle(clientID, visionSensorName,
                                                              vrep_sim.simx_opmode_blocking)
    if (return_code == vrep_sim.simx_return_ok):
        print('get camera handle ok.')

        errprCode, resolution, image = vrep_sim.simxGetVisionSensorImage(clientID, camera_handle, 0,
                                                                         vrep_sim.simx_opmode_blocking)
        print("image_buffer: ", len(image))
        sensorFrame = np.array(image, dtype=np.uint8)
        sensorFrame.resize([resolution[0], resolution[1], 3])  # 调整图像通道结构
        cv2.imshow('', sensorFrame)
        cv2.waitKey(0)

    return 0

def getVisionSensorRGBDepthData(vrep_sim, clientID, visionSensorName):
    return_code, camera_handle = vrep_sim.simxGetObjectHandle(clientID, visionSensorName,
                                                              vrep_sim.simx_opmode_oneshot_wait)
    if (return_code == vrep_sim.simx_return_ok):
        print('get camera handle ok.')

        errprCode, resolution, image = vrep_sim.simxGetVisionSensorDepthBuffer(clientID, camera_handle, vrep_sim.simx_opmode_blocking)
        print("image_buffer: ", len(image))
        sensorFrame = np.array(image, dtype=np.uint8)
        sensorFrame.resize([resolution[0], resolution[1]])  # 调整图像通道结构
        cv2.imshow('', sensorFrame)
        cv2.waitKey(0)

    return 0


if __name__ == '__main__':
    try:
        import sim as vrep_sim
    except:
        print ('--------------------------------------------------------------')
        print ('"sim.py" could not be imported. This means very probably that')
        print ('either "sim.py" or the remoteApi library could not be found.')
        print ('Make sure both are in the same folder as this file,')
        print ('or appropriately adjust the file "sim.py"')
        print ('--------------------------------------------------------------')
        print ('')

    ''' Initialization '''
    print('Program started')
    vrep_sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = vrep_sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim

    if clientID != -1:
        print('Connected to remote API server')

        ObjName = 'camera'
        obj_pos, obj_ori = getObjectPosition(vrep_sim, clientID, ObjName)
        print('{}: position is {}, orientation is {}.'.format(ObjName,obj_pos, obj_ori))

        print('getting vision sensor data...')
        getVisionSensorRGBData(vrep_sim, clientID, ObjName)

    else:
        print('Failed connecting to remote API server')

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    vrep_sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    vrep_sim.simxFinish(clientID)
    print('Program ended')