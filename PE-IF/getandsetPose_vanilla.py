import time
import sys
import numpy as np
import cv2
sys.path.append('./MODEL_pythonAPI/VREP_remoteAPIs')

if __name__ == '__main__':
    try:
        import sim as vrep_sim
    except:
        print('--------------------------------------------------------------')
        print('"sim.py" could not be imported. This means very probably that')
        print('either "sim.py" or the remoteApi library could not be found.')
        print('Make sure both are in the same folder as this file,')
        print('or appropriately adjust the file "sim.py"')
        print('--------------------------------------------------------------')
        print('')

    ''' Initialization '''
    print('Program started')
    vrep_sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = vrep_sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim
    if clientID != -1:
        print('Connected to remote API server')

        # get the pose of note11
        return_code, note11_handle = vrep_sim.simxGetObjectHandle(clientID, 'note11', vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print('get object note11 ok.')

            _, note11_pos = vrep_sim.simxGetObjectPosition(clientID, note11_handle, -1, vrep_sim.simx_opmode_blocking)
            _, note11_ori = vrep_sim.simxGetObjectOrientation(clientID, note11_handle, -1, vrep_sim.simx_opmode_blocking)
            print("The position of note11: ", note11_pos)
            print("The orientation of note11: ", note11_ori)

        # get the pose of vision sensor and visualize video data
        return_code, visionsensor_handle = vrep_sim.simxGetObjectHandle(clientID, 'cammask',
                                                                     vrep_sim.simx_opmode_oneshot_wait)
        if (return_code == vrep_sim.simx_return_ok):
            print('get object Realsense ok.')

            _, visionsensor_pos = vrep_sim.simxGetObjectPosition(clientID, visionsensor_handle, -1,
                                                              vrep_sim.simx_opmode_streaming)
            _, visionsensor_ori = vrep_sim.simxGetObjectOrientation(clientID, visionsensor_handle, -1,
                                                                 vrep_sim.simx_opmode_streaming)

            errprCode, resolution, image = vrep_sim.simxGetVisionSensorImage(clientID, visionsensor_handle, 0,
                                                                             vrep_sim.simx_opmode_streaming)

            time.sleep(0.5)
            while True:
                _, visionsensor_pos = vrep_sim.simxGetObjectPosition(clientID, visionsensor_handle, -1,
                                                                  vrep_sim.simx_opmode_buffer)
                _, visionsensor_ori = vrep_sim.simxGetObjectOrientation(clientID, visionsensor_handle, -1,
                                                                     vrep_sim.simx_opmode_buffer)
                print("The position of vision sensor: ", visionsensor_pos)
                print("The Orientation of vision sensor: ", visionsensor_pos)

                errprCode, resolution, image = vrep_sim.simxGetVisionSensorImage(clientID, visionsensor_handle, 0,
                                                                                 vrep_sim.simx_opmode_buffer)

                sensorFrame = np.array(image, dtype=np.uint8)
                sensorFrame.resize([resolution[0], resolution[1], 3])  # 调整图像通道结构
                cv2.imshow('', sensorFrame)
                key = cv2.waitKey(1)
                if int(key) == 27:
                    # 通过esc键退出摄像
                    cv2.destroyAllWindows()
                    break


    else:
        print('Failed connecting to remote API server')