import time
import os
import csv
import copy

import cv2
import pyro
import pyro.distributions as dist
import torch
import numpy as np
from tqdm import tqdm
import transforms3d as tf3d

from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder

from generation.mujocoCabinetParts import build_cabinet, sample_cabinet
from generation.mujocoDrawerParts import build_drawer, sample_drawers
from generation.mujocoMicrowaveParts import build_microwave, sample_microwave
from generation.mujocoToasterOvenParts import build_toaster, sample_toaster
from generation.mujocoDoubleCabinetParts import build_cabinet2, sample_cabinet2, set_two_door_control
from generation.mujocoRefrigeratorParts import build_refrigerator, sample_refrigerator
from generation.utils import *
import generation.calibrations as calibrations

def white_bg(img):
    mask = 1 - (img > 0)
    img_cp = copy.deepcopy(img)
    img_cp[mask.all(axis=2)] = [255,255,255]
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

        else:
            raise 'uh oh, object not implemented!'
        return obj

    def generate_scenes(self, N, objtype, write_csv=True, save_imgs=True, mean_flag=False, left_only=False, cute_flag=False, test=False, video=False):
        fname=os.path.join(self.savedir, 'params.csv')
        self.img_idx = 0
        with open(fname, 'a') as csvfile:
            writ = csv.writer(csvfile, delimiter=',')
            for i in tqdm(range(N)):
                obj = self.sample_obj(objtype, mean_flag, left_only, cute_flag=cute_flag)
                xml=obj.xml
                fname=os.path.join(self.savedir, 'scene'+str(i).zfill(6)+'.xml')
                self.write_urdf(fname, xml)
                self.scenes.append(fname)
                self.take_images(fname, obj, writ)
        return

    def take_images(self, filename, obj, writer, img_idx=0, debug=False):
        model = load_model_from_path(filename)
        sim = MjSim(model)
        modder= TextureModder(sim)
        # viewer=MjViewer(sim) # this fucking line has caused me so much pain.

        embedding = np.append(obj.type, obj.geom.reshape(-1))
        if obj.type == 4 or obj.type == 5:
            # MULTI CABINET: get double the params.
            axis1, door1 = transform_param(obj.params[0][0],obj.params[0][1], obj)
            axis2, door2 = transform_param(obj.params[1][0],obj.params[1][1], obj)
            axes = np.append(axis1,axis2)
            doors = np.append(door1,door2)
            params = np.append(axes,doors)
            set_two_door_control(sim, 'cabinet2' if obj.type == 4 else 'refrigerator')
            n_qpos_variables = 2

        else:
            n_qpos_variables = 1
            if obj.type == 1:
                sim.data.ctrl[0] = 0.05

            elif obj.geom[3] == 1:
                sim.data.ctrl[0] = -0.2

            else:
                sim.data.ctrl[0] = 0.2

            params = get_cam_relative_params2(obj) # if 1DoF, params is length 10. If 2DoF, params is length 20.

        embedding_and_params = np.concatenate((embedding, params, obj.pose, obj.rotation))
        # print('nqpos', n_qpos_variables)
        # print(self.img_idx, obj.pose)
        # print(embedding_and_params.shape)
        t = 0
        while t<4000:
            sim.forward()
            sim.step()

            if t%250==0:

                #########################
                IMG_WIDTH = calibrations.sim_width
                IMG_HEIGHT = calibrations.sim_height
                #########################

                img, depth = sim.render(IMG_WIDTH, IMG_HEIGHT, camera_name='external_camera_0', depth=True)
                depth = vertical_flip(depth)
                real_depth = buffer_to_real(depth, 12.0, 0.1)
                norm_depth = real_depth / 12.0

                if self.masked:
                    # remove background
                    mask = norm_depth > 0.99
                    norm_depth = (1-mask)*norm_depth

                if self.debugging:
                    # save image to disk for visualization
                    # img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))

                    img = vertical_flip(img)
                    img = white_bg(img)
                    imgfname = os.path.join(self.savedir, 'img'+str(self.img_idx).zfill(6)+'.png')
                    depth_imgfname = os.path.join(self.savedir, 'depth_img'+str(self.img_idx).zfill(6)+'.png')
                    integer_depth = norm_depth * 255
                    cv2.imwrite(imgfname, img)
                    cv2.imwrite(depth_imgfname, integer_depth)

                # if IMG_WIDTH != 192 or IMG_HEIGHT != 108:
                #     depth = cv2.resize(norm_depth, (192,108))

                depthfname = os.path.join(self.savedir,'depth'+str(self.img_idx).zfill(6) + '.pt')
                torch.save(torch.tensor(norm_depth.copy()), depthfname)
                row = np.append(embedding_and_params, sim.data.qpos[:n_qpos_variables])
                # print(row.shape)
                writer.writerow(row)
                self.img_idx += 1

            t += 1

# shapes and stuff
# if 1DoF, params is length 10. If 2DoF, params is length 20.
# embedding is length 5: type, l, w, h, left
# pose is length 3
# rotation is length 4
# finally, q is length 1 or 2.
# thus, for generating shape data:
# 1DoF: q is position 21
# 2DoF: q is position 31

# Object IDs
# 0 - microwave
# 1 - drawer
# 2 - cabinet
# 3 - toaster
# 4 - double cabinet
# 5 - refrigerator
