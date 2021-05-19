import time
import os
import csv
import copy
import math
import pickle

import cv2
import pyro
import pyro.distributions as dist
import torch
import numpy as np
from tqdm import tqdm
import transforms3d as tf3d

import pybullet as pb
import random

from generation.mujocoCabinetParts import build_cabinet, sample_cabinet
from generation.mujocoDrawerParts import build_drawer, sample_drawers
from generation.mujocoMicrowaveParts import build_microwave, sample_microwave
from generation.mujocoToasterOvenParts import build_toaster, sample_toaster
from generation.mujocoDoubleCabinetParts import build_cabinet2, sample_cabinet2, set_two_door_control
from generation.mujocoRefrigeratorParts import build_refrigerator, sample_refrigerator
from generation.utils import *
import generation.calibrations as calibrations

pb_client = pb.connect(pb.GUI)
pb.setGravity(0,0,-100)

def white_bg(img):
    mask = 1 - (img > 0)
    img_cp = copy.deepcopy(img)
    img_cp[mask.all(axis=2)] = [255,255,255, 0]
    return img_cp

def buffer_to_real(z, zfar, znear):
    return 2*zfar*znear / (zfar + znear - (zfar - znear)*(2*z -1))

def vertical_flip(img):
    return np.flip(img, axis=0)

class SceneGenerator():
    def __init__(self, root_dir='bull/test_cabinets/solo', masked=False, debug_flag=False):
        '''
        Class for generating simulated articulated object dataset.
        params:
            - root_dir: save in this directory
            - start_idx: index of first image saved - useful in threading context
            - depth_data: np array of depth images
            - masked: should the background of depth images be 0s or 1s?
        '''
        self.scenes=[]
        self.savedir=root_dir
        self.masked = masked
        self.img_idx = 0
        self.depth_data=[]
        self.debugging=debug_flag

        # Camera external settings
        self.viewMatrix = pb.computeViewMatrix(
            cameraEyePosition=[4,0,1],
            cameraTargetPosition=[0,0,1],
            cameraUpVector=[0,0,1]
        )

        # Camera internal settings
        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov=45.,
            aspect=1.0,
            nearVal=0.1,
            farVal=8.1
        )

        print(root_dir)

    def write_urdf(self, filename, xml):
        with open(filename, "w") as text_file:
            text_file.write(xml)

    def sample_obj(self, obj_type, mean_flag, left_only, cute_flag=False):
        if obj_type == 'microwave':
            l, w, h, t, left, mass = sample_microwave(mean_flag)
            if mean_flag:
                obj = build_microwave(l, w, h, t, left,
                                      set_pose = [1.0, 0.0, -0.15],
                                      set_rot = [0.0, 0.0, 0.0, 1.0] )
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_microwave(l, w, h, t, left,
                                      set_pose = [1.0, 0.0, -0.15],
                                      set_rot = base_quat)
            else:
                obj = build_microwave(l, w, h, t, left)

            camera_dist = max(1.25, 2*math.log(10*h))
            camera_height = h/2.

        elif obj_type == 'drawer':
            l, w, h, t, left, mass = sample_drawers(mean_flag)
            if mean_flag:
                obj = build_drawer(l, w, h, t, left,
                                   set_pose = [1.5, 0.0, -0.4],
                                   set_rot = [0.0, 0.0, 0.0, 1.0] )
            elif cute_flag:
               base_xyz, base_angle = sample_pose()
               base_quat = angle_to_quat(base_angle)
               obj = build_drawer(l, w, h, t, left,
                                     set_pose = [1.2, 0.0, -0.15],
                                     set_rot = base_quat)
            else:
                obj = build_drawer(l, w, h, t, left)

            camera_dist = max(2, 2*math.log(10*h))
            camera_height = h/2.

        elif obj_type == 'toaster':
            l, w, h, t, left, mass = sample_toaster(mean_flag)
            if mean_flag:
                obj = build_toaster(l, w, h, t, left,
                                    set_pose = [1.5, 0.0, -0.3],
                                    set_rot = [0.0, 0.0, 0.0, 1.0] )
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_toaster(l, w, h, t, left,
                                      set_pose = [1.0, 0.0, -0.15],
                                      set_rot = base_quat)
            else:
                obj = build_toaster(l, w, h, t, left)

            camera_dist = max(1, 2*math.log(10*h))
            camera_height = h/2.

        elif obj_type == 'cabinet':
            l, w, h, t, left, mass = sample_cabinet(mean_flag)
            if mean_flag:
                if left_only:
                    left=True
                else:
                    left=False
                obj = build_cabinet(l, w, h, t, left,
                                    set_pose = [1.5, 0.0, -0.3],
                                    set_rot = [0.0, 0.0, 0.0, 1.0] )
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_cabinet(l, w, h, t, left,
                                      set_pose = [1.5, 0.0, -0.15],
                                      set_rot = base_quat)
            else:
                left = np.random.choice([True,False])
                obj = build_cabinet(l, w, h, t, left)
            
            camera_dist = 2*math.log(10*h)
            camera_height = h/2.

        elif obj_type == 'cabinet2':
            l, w, h, t, left, mass = sample_cabinet2(mean_flag)
            if mean_flag:
                obj = build_cabinet2(l, w, h, t, left,
                                     set_pose = [1.5, 0.0, -0.3],
                                     set_rot = [0.0, 0.0, 0.0, 1.0] )
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_cabinet2(l, w, h, t, left,
                                      set_pose = [1.5, 0.0, -0.15],
                                      set_rot = base_quat)
            else:
                obj = build_cabinet2(l, w, h, t, left)

            camera_dist = 2*math.log(10*h)
            camera_height = h/2.

        elif obj_type == 'refrigerator':
            l, w, h, t, left, mass = sample_refrigerator(mean_flag)
            if mean_flag:

                obj = build_refrigerator(l, w, h, t, left,
                                         set_pose = [1.5, 0.0, -0.3],
                                         set_rot = [0.0, 0.0, 0.0, 1.0])
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_refrigerator(l, w, h, t, left,
                                      set_pose = [2.5, 0.0, -0.75],
                                      set_rot = base_quat)

            else:
                obj = build_refrigerator(l, w, h, t, left)

            camera_dist = 2*math.log(10*h)
            camera_height = h/2.

        else:
            raise 'uh oh, object not implemented!'
        return obj, camera_dist, camera_height

    def generate_scenes(self, N, objtype, write_csv=True, save_imgs=True, mean_flag=False, left_only=False, cute_flag=False, test=False, video=False):
        fname=os.path.join(self.savedir, 'labels.csv')
        self.img_idx = 0
        with open(fname, 'a') as csvfile:
            writ = csv.writer(csvfile, delimiter=',')
            writ.writerow(['Object Name', 'Joint Type', 'Image Index', 'l_1', 'l_2', 'l_3', 'm_1', 'm_2', 'm_3',])
            for i in tqdm(range(N)):
                obj, camera_dist, camera_height = self.sample_obj(objtype, mean_flag, left_only, cute_flag=cute_flag)
                xml=obj.xml
                fname=os.path.join(self.savedir, 'scene'+str(i).zfill(6)+'.xml')
                self.write_urdf(fname, xml)
                self.scenes.append(fname)
                self.take_images(fname, obj, camera_dist, camera_height, obj.joint_index, writ, test=test, video=video)
        return


    def take_images(self, filename, obj, camera_dist, camera_height, joint_index, writer, img_idx=0, debug=False, test=False, video=False):
        
        objId, _ = pb.loadMJCF(filename)

        # create texture image
        x, y = np.meshgrid(np.linspace(-1,1, 128), np.linspace(-1,1, 128))
        texture_img = (72*(np.stack([np.cos(16*x), np.cos(16*y), np.cos(16*(x+y))])+2)).astype(np.uint8).transpose(1,2,0)
        from PIL import Image
        texture_img = Image.fromarray(texture_img)
        fname = 'texture_test.png'
        texture_img.save(fname)
        textureId = pb.loadTexture(fname, physicsClientId=pb_client)

        # apply texture to the object way: idea one
        # planeVis = pb.createVisualShape(shapeType=pb.GEOM_MESH,
        #                        fileName=filename,
        #                        rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0], 
        #                        specularColor=[0.5, 0.5, 0.5],
        #                        physicsClientId=pb_client)

        # pb.changeVisualShape(planeVis,
        #                     -1,
        #                     textureUniqueId=textureId,
        #                     rgbaColor=[1, 1, 1, 1],
        #                     specularColor=[1, 1, 1, 1],
        #                     physicsClientId=pb_client)

        # apply texture to the object way: idea two
        # pb.changeVisualShape(objId, -1, textureUniqueId=textureId, rgbaColor=[1, 1, 1, 1], specularColor=[1, 1, 1, 1], physicsClientId=pb_client) #bottom 
        # pb.changeVisualShape(objId, 0, textureUniqueId=textureId, rgbaColor=[1, 1, 1, 1], specularColor=[1, 1, 1, 1], physicsClientId=pb_client) #left side
        # pb.changeVisualShape(objId, 1, textureUniqueId=textureId, rgbaColor=[1, 1, 1, 1], specularColor=[1, 1, 1, 1], physicsClientId=pb_client) #right side

        # change visual shape without the texture add
        # pb.changeVisualShape(objId, -1, rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0], specularColor=[0.5, 0.5, 0.5]) 

        self.viewMatrix = pb.computeViewMatrix(
            cameraEyePosition=[camera_dist,0,1],
            cameraTargetPosition=[0,0,camera_height],
            cameraUpVector=[-1,0,1]
        )
        
        # Take 16 pictures, permuting orientation and joint extension
        for t in range(1,4000):
            pb.stepSimulation()

            if t%250==0:
                state = {'startPos': [0,0,0], 'eulerOrientation': [], 'joints': [], 'joint_index': joint_index, 'img_idx': self.img_idx}

                # Randomly update orientation to new orientation between -pi/4 and pi/4
                startPos = [0,0,0]
                obj_rotation = np.random.uniform(-np.pi/4.,np.pi/4.)
                startOrientation = pb.getQuaternionFromEuler([0,0,obj_rotation])
                state['eulerOrientation'] = [0,0,obj_rotation]
                pb.resetBasePositionAndOrientation(objId, startPos, startOrientation)
                
                # Update joint extension randomly between 0 and 120
                for j in range(pb.getNumJoints(objId)):
                    if obj.control[j] < 0:
                        rotation = np.random.uniform(obj.control[j], 0)
                    else: 
                        rotation = np.random.uniform(0, obj.control[j])
                    state['joints'].append(rotation)
                    pb.resetJointState(objId, j, rotation)

                #########################
                IMG_WIDTH = calibrations.sim_width
                IMG_HEIGHT = calibrations.sim_height
                #########################

                # # Take picture without texture
                # width, height, img, depth, segImg = pb.getCameraImage(
                #     IMG_WIDTH, # width
                #     IMG_HEIGHT, # height
                #     self.viewMatrix,
                #     self.projectionMatrix, 
                #     lightDirection=[camera_dist, 0, camera_height+1], # light source
                #     shadow=1, # include shadows
                # )

                # use projective texture, it's more robust, applies texture on all sides at once
                viewMat = [
                    0.642787516117096, -0.4393851161003113, 0.6275069713592529, 0.0, 0.766044557094574,
                    0.36868777871131897, -0.5265407562255859, 0.0, -0.0, 0.8191521167755127, 0.5735764503479004,
                    0.0, 2.384185791015625e-07, 2.384185791015625e-07, -5.000000476837158, 1.0
                ]
                projMat = [
                    0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0,
                    0.0, 0.0, -0.02000020071864128, 0.0
                ]
                
                width, height, img, depth, segImg = pb.getCameraImage(
                                        IMG_WIDTH, # width
                                        IMG_HEIGHT, # height
                                        self.viewMatrix,
                                        self.projectionMatrix,
                                        renderer=pb.ER_BULLET_HARDWARE_OPENGL,
                                        flags=pb.ER_USE_PROJECTIVE_TEXTURE,
                                        projectiveTextureView=viewMat,
                                        projectiveTextureProj=projMat)


                if test:
                    state['viewMatrix'] = self.viewMatrix
                    state['projectionMatrix'] = self.projectionMatrix
                    state['lightDirection'] = [camera_dist, 0, camera_height+1]
                    state['height'] = IMG_HEIGHT 
                    state['width'] = IMG_WIDTH 
                    state['mjcf'] = filename

                    config_name = os.path.join(self.savedir,'config'+str(self.img_idx).zfill(6)+'.pkl')
                    f = open(config_name,"wb")
                    pickle.dump(state,f)
                    f.close()
                
                #depth = vertical_flip(depth)
                real_depth = buffer_to_real(depth, 12.0, 0.1)
                norm_depth = real_depth / 12.0

                if self.masked:
                    # remove background
                    mask = norm_depth > 0.99
                    norm_depth = (1-mask)*norm_depth

                if self.debugging:
                    # save image to disk for visualization
                    # img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))

                    #img = vertical_flip(img)

                    img = white_bg(img)
                    imgfname = os.path.join(self.savedir, 'img'+str(self.img_idx).zfill(6)+'.png')
                    depth_imgfname = os.path.join(self.savedir, 'depth_img'+str(self.img_idx).zfill(6)+'.png')
                    integer_depth = norm_depth * 255
                    cv2.imwrite(imgfname, img)
                    cv2.imwrite(depth_imgfname, integer_depth)

                # if IMG_WIDTH != 192 or IMG_HEIGHT != 108:
                #     depth = cv2.resize(norm_depth, (192,108))

                if joint_index is None:
                    raise Exception("Joint index not defined! Are you simulating a 2DOF object? (Don't do that yet)")

                large_door_joint_info = pb.getJointInfo(objId, joint_index)
                p = np.array(list(large_door_joint_info[14]))
                l = np.array(list(large_door_joint_info[13]))
                m = np.cross(large_door_joint_info[14], large_door_joint_info[13])

                depthfname = os.path.join(self.savedir,'depth'+str(self.img_idx).zfill(6) + '.pt')
                torch.save(torch.tensor(norm_depth.copy()), depthfname)
                row = np.concatenate((np.array([obj.name, obj.joint_type, self.img_idx]),l, m)) # SAVE SCREW REPRESENTATION HERE 
                writer.writerow(row)

                if video:
                    increments = {j: 0 for j in range(pb.getNumJoints(objId))}
                    videoFolderFname = os.path.join(self.savedir, 'video_for_img_'+str(self.img_idx).zfill(6))
                    os.makedirs(videoFolderFname, exist_ok=False)
                    for frame_idx in range(90):
                        for j in range(pb.getNumJoints(objId)):
                            pb.resetJointState(objId, j, increments[j])
                            increments[j] += obj.control[j]/90
                        
                        _, _, rgbFrame, depthFrame, _ = pb.getCameraImage(
                            IMG_WIDTH, # width
                            IMG_HEIGHT, # height
                            self.viewMatrix,
                            self.projectionMatrix, 
                            lightDirection=[camera_dist, 0, camera_height+1], # light source
                            shadow=1, # include shadows
                        )

                        frameRgbFname = os.path.join(videoFolderFname, 'rgb_frame_'+str(frame_idx).zfill(6)+'.png')
                        rgbFrame = white_bg(rgbFrame)
                        cv2.imwrite(frameRgbFname, rgbFrame)
                        frameDepthFname = os.path.join(videoFolderFname, 'depth_frame_'+str(frame_idx).zfill(6)+'.pt')
                        real_depth = buffer_to_real(depthFrame, 12.0, 0.1)
                        norm_depth = real_depth / 12.0

                        if self.masked:
                            # remove background
                            mask = norm_depth > 0.99
                            norm_depth = (1-mask)*norm_depth

                        torch.save(torch.tensor(norm_depth.copy()), frameDepthFname)

                self.img_idx += 1
