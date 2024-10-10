import time
import math
import sys
sys.path.append('./MODEL_pythonAPI/VREP_remoteAPIs')


''' Arm control function '''
def VREP_armControl(sim, clientID, arm_joints_handle, desired_arm_joint_angles):
    jointsNum = len(arm_joints_handle)
    for i in range(0,jointsNum):
        sim.simxSetJointPosition(clientID, arm_joints_handle[i], desired_arm_joint_angles[i], sim.simx_opmode_blocking)

if __name__ == '__main__':
    try:
        import sim
    except:
        print ('--------------------------------------------------------------')
        print ('"sim.py" could not be imported. This means very probably that')
        print ('either "sim.py" or the remoteApi library could not be found.')
        print ('Make sure both are in the same folder as this file,')
        print ('or appropriately adjust the file "sim.py"')
        print ('--------------------------------------------------------------')
        print ('')

    ''' Initialization '''
    print ('Program started')
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
    if clientID != -1:
        print ('Connected to remote API server')

        # Prepare initial values for five arm joints
        arm_joints_handle = [-1, -1, -1, -1, -1, -1]
        for i in range(0, 6):
            return_code, arm_joints_handle[i] = sim.simxGetObjectHandle(clientID, 'elfin_joint' + str(i+1),
                                                                             sim.simx_opmode_blocking)
            if (return_code == sim.simx_return_ok):
                print('get object arm joint ' + str(i+1) + ' ok.')
                _, arm_joint_pos = sim.simxGetObjectPosition(clientID, arm_joints_handle[i], -1,
                                                                  sim.simx_opmode_blocking)
                print('get arm joint position: ', arm_joint_pos)

        # Desired joint positions for initialization
        desired_arm_joint_angles = [90 * math.pi / 180, -20 * math.pi / 180, -100 * math.pi / 180,
                                    0, -60 * math.pi / 180, 0]

        # Initialization all arm joints
        for i in range(0, 6):
            sim.simxSetJointPosition(clientID, arm_joints_handle[i], desired_arm_joint_angles[i],
                                          sim.simx_opmode_blocking)
            print('set arm joint position ' + str(i+1) + ' ok.')


         # User variables
        simu_time = 0

        ''' Main control loop '''
        print('begin main control loop ...')
        while True:
            # Motion planning
            simu_time = simu_time + 0.05

            for i in range(0, 6):
                if int(simu_time) % 2 == 0:
                    desired_arm_joint_angles[i] = desired_arm_joint_angles[i] - 0.04  # rad
                else:
                    desired_arm_joint_angles[i] = desired_arm_joint_angles[i] + 0.04  # rad

            # Control the arm
            VREP_armControl(sim, clientID, arm_joints_handle, desired_arm_joint_angles)
    else:
        print('Failed connecting to remote API server')



    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
    print('Program ended')