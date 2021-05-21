'''
THIS FILE IS FOR SANDBOXING AND DEBUGGING. PLEASE IGNORE.
'''

import os
import pybullet as p
import time
import pybullet_data
import random
import argparse
import numpy as np
from PIL import Image

# Useful resource: https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
planeId = p.loadURDF("plane.urdf")
p.setGravity(0,0,-100)

def generate_images_from_mjcf(fpath):
    objId, _ = p.loadMJCF(fpath)

    # Initialize object pose
    startPos = [0,0,1]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    p.resetBasePositionAndOrientation(objId, startPos, startOrientation)

    # Camera external settings
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[4,0,2],
        cameraTargetPosition=[0,0,1],
        cameraUpVector=[-1,0,1]
    )

    # Camera internal settings
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=45.,
        aspect=1.0,
        nearVal=0.1,
        farVal=8.1
    )

    large_door_joint_info = p.getJointInfo(objId, 5)
    print(large_door_joint_info[12])

    # Permute color of object components slightly
    #for i in range(p.getNumJoints(objId)):
    #    noise = random.gauss(0, 0.1)
    #    p.changeVisualShape(objId, i, rgbaColor=[105.+noise,105.+noise, 105.+noise, 1.])
    #p.changeVisualShape(objId, -1, rgbaColor=[105.+noise,105.+noise, 105.+noise, 1.])

    # Render object initialization
    for i in range(20):
        p.stepSimulation()

        '''
        if i % 100 == 0:
            width, height, rbgImg, depthImg, segImg = p.getCameraImage(
                500,
                500, 
                viewMatrix,
                projectionMatrix, 
                lightDirection=[4,-4,2],
                shadow=1,
            )
        '''
        #print(i)
        #p.resetJointState(objId, 5, 2)
        #time.sleep(1.)

    # Take 10 pictures, permuting orientation and joint extension
    for i in range(10):
        # Randomly update orientation to new orientation between -pi/4 and pi/4
        startPos = [0,0,0]
        rotation = np.random.uniform(-np.pi/4.,np.pi/4.)
        startOrientation = p.getQuaternionFromEuler([0,0,rotation])
        p.resetBasePositionAndOrientation(objId, startPos, startOrientation)
        
        # Update joint extension radomly between 0 and 120
        # Brute force update of all joints (Ideally, we update this 
        # to only update the interactive joints, but this is fine for all practical purposes.)
        for j in range(p.getNumJoints(objId)):
            rotation = np.random.uniform(0, np.pi/2.)
            p.resetJointState(objId, j, rotation)

        # Take picture
        width, height, rbgImg, depthImg, segImg = p.getCameraImage(
                500, # width
                500, # height
                viewMatrix,
                projectionMatrix, 
                lightDirection=[4,-4,2], # light source
                shadow=1, # include shadows
            )

        # Save RGB picture
        img = np.uint8(np.array(rbgImg))
        img = Image.fromarray(img)
        #img.save('./rgb_images/'+current_obj+'/'+current_obj+str(i)+'.png')

        # Wait a little
        time.sleep(1.)
        
        '''
        large_door_joint_info = p.getJointInfo(objId, 4)
        print('Point p on joint axis: ', large_door_joint_info[14])
        print('Direction l: ', large_door_joint_info[13])
        m = np.cross(large_door_joint_info[14], large_door_joint_info[13])
        print('Momentum vector m: ', m)
        '''
        


def main(args):
   objects = ['microwave', 'drawer', 'toaster', 'cabinet', 'cabinet2', 'refrigerator'] 

# TODO: Generate all images here, and then get the names of all of them, and iterate below code for each of them
objects = ['microwave', 'drawer', 'toaster', 'cabinet', 'cabinet2', 'refrigerator']
current_obj = objects[1]

#objId, _ = p.loadMJCF('./generated_mjcf/'+current_obj+'/scene000000.xml')


generate_images_from_mjcf('./generated_mjcf/'+current_obj+'/scene000000.xml')
