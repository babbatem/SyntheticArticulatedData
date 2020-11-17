import os
import pybullet as p
import time
import pybullet_data
import random
import numpy as np
from PIL import Image

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
p.setGravity(0,0,-100)
planeId = p.loadURDF("plane.urdf")
#cubeStartPos = [0,0,1]
#cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
#boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)

import inspect 

#cubeStartPos = [0,0,1]
#cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

objects = ['microwave', 'drawer', 'toaster', 'cabinet', 'cabinet2', 'refrigerator']

testId = p.loadMJCF('./generation/test/'+objects[5]+'/scene000000.xml')


print(testId)

startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
p.resetBasePositionAndOrientation(testId[0], startPos, startOrientation)

viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[4,0,1],
    cameraTargetPosition=[0,0,1],
    cameraUpVector=[0,0,1]
)

projectionMatrix = p.computeProjectionMatrixFOV(
    fov=45.,
    aspect=1.0,
    nearVal=0.1,
    farVal=8.1
)

jointPos, jointVol, _, _ = p.getJointState(testId[0], 0)
print(p.getNumJoints(testId[0]))
for i in range(p.getNumJoints(testId[0])):
    p.resetJointState(testId[0],p.getNumJoints(testId[0])-4, 90)

for i in range(p.getNumJoints(testId[0])):
    noise = random.gauss(0, 0.1)
    p.changeVisualShape(testId[0], i, rgbaColor=[105.+noise,105.+noise, 105.+noise, 1.])
p.changeVisualShape(testId[0], -1, rgbaColor=[105.+noise,105.+noise, 105.+noise, 1.])

for i in range(20):
    p.stepSimulation()

    if i % 100 == 0:
        width, height, rbgImg, depthImg, segImg = p.getCameraImage(
            500,
            500, 
            viewMatrix,
            projectionMatrix, 
            lightDirection=[4,-4,2],
            shadow=1,
        )
    
    time.sleep(1./2500.)

# Take a variety of pictures
for i in range(20):

    # Randomly update orientation to new orientation between -pi/4 and pi/4
    startPos = [0,0,0]
    rotation = np.random.uniform(-np.pi/4.,np.pi/4.)
    startOrientation = p.getQuaternionFromEuler([0,0,rotation])
    p.resetBasePositionAndOrientation(testId[0], startPos, startOrientation)
    
    # Update joint extension radomly between 0 and 120
    for j in range(p.getNumJoints(testId[0])):
        rotation = np.random.uniform(0, np.pi/2.)
        p.resetJointState(testId[0],j, rotation)

    # Take picture
    width, height, rbgImg, depthImg, segImg = p.getCameraImage(
            500,
            500, 
            viewMatrix,
            projectionMatrix, 
            lightDirection=[4,-4,2],
            shadow=1,
        )

    print(np.array(depthImg).shape)
    # Save picture
    img = np.uint8(np.array(rbgImg))
    img = Image.fromarray(img)
    #img.save('./refrigerator_images_demo/refrigerator'+str(i)+'.png')

    # Wait a little
    time.sleep(1.)    

# https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd


while True:
    p.stepSimulation()