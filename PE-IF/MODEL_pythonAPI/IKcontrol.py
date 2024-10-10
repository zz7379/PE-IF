import sys
sys.path.append('./MODEL_pythonAPI/VREP_remoteAPIs')
def armIKcontrol(vrep_sim, clientID, targetName, targetPosition):
    return_code, target_handle = vrep_sim.simxGetObjectHandle(clientID, targetName,
                                                              vrep_sim.simx_opmode_blocking)
    if (return_code == vrep_sim.simx_return_ok):
        print('get target handle ok.')
        vrep_sim.simxSetObjectPosition(clientID, target_handle, -1, targetPosition,
                                       vrep_sim.simx_opmode_blocking)
        print('set target position ok.')
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
    print ('Program started')
    vrep_sim.simxFinish(-1) # just in case, close all opened connections
    clientID = vrep_sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim

    if clientID != -1:
        print ('Connected to remote API server')

        targetName = 'elfin5_target'
        #targetPosition = [-2.38928,-0.00,1.34483]
        targetPosition = [-2.69599,0.24874,1.40116]
        armIKcontrol(vrep_sim, clientID, targetName, targetPosition)

    else:
        print('Failed connecting to remote API server')

     # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    vrep_sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    vrep_sim.simxFinish(clientID)
    print('Program ended')