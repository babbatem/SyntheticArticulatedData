import numpy as np
import pyro
import pyro.distributions as dist
import torch

from generation.ArticulatedObjs import Drawer, ArticulatedObject
from generation.utils import *

d_len = dist.Uniform(18/2*0.0254, 24/2*0.0254)
d_width = dist.Uniform(12/2*0.0254, 30/2*0.0254)
d_height = dist.Uniform(4/2*0.0254, 12/2*0.0254)
d_thicc = dist.Uniform(0.01 / 2, 0.05 / 2)

def get_fake_pose():
    base_xyz = [3.0,0.0,0.0]
    base_angle= 0.0
    return tuple(base_xyz), base_angle

def sample_drawers(mean_flag):
    if mean_flag:
        print('generating mean drawer')
        length = d_len.mean
        width = d_width.mean
        height = d_height.mean
        thickness=d_thicc.mean
    else:
        length=pyro.sample("length", d_len).item() #aka depth
        width =pyro.sample('width', d_width).item()
        height=pyro.sample('height', d_height).item()
        thickness=pyro.sample('thicc', d_thicc).item()
    left=0
    mass=5.0
    return length, width, height, thickness, left, mass

def build_drawer(length, width, height, thicc, left, set_pose=None, set_rot=None):
    base_length=length
    base_width=width
    base_height=thicc

    if set_pose is None:
        base_xyz, base_angle = sample_pose_drawer()
        base_quat = angle_to_quat(base_angle)
    else:
        base_xyz = tuple(set_pose)
        base_quat = tuple(set_rot)


    base_origin=make_string(base_xyz)
    base_orientation = make_quat_string(base_quat)

    base_size = make_string((base_length, base_width, base_height))
    side_length=length
    side_width=thicc
    side_height=height
    side_size = make_string((side_length, side_width, side_height))

    back_size = make_string((side_width, base_width, side_height))
    top_size = base_size
    door_size = make_string((side_width, base_width * 3 / 4, side_height))

    left_origin  = make_string((0, -width + thicc, height))
    right_origin = make_string((0, width - thicc, height))
    top_origin = make_string((0,0,height*2))
    back_origin = make_string((-base_length + thicc, 0.0, height))

    # # TODO: sample drawer height and width
    drawer_bottom_origin = make_string((0, 0, thicc / 2))
    drawer_left_origin  = make_string((0, -width + 2*thicc, height))
    drawer_right_origin = make_string((0, width - 2*thicc, height))
    drawer_front_origin = make_string((base_length - 2*thicc, 0.0, height))
    drawer_back_origin = make_string((-base_length + thicc, 0.0, height))

    drawer_bottom_size = make_string((base_length - thicc, base_width - thicc, base_height))
    drawer_back_size = make_string((side_width, base_width - thicc, side_height - thicc))
    drawer_side_size = make_string((side_length - thicc, side_width, side_height - thicc))

    # drawer_height = pyro.sample("drawer_height", dist.Uniform(2.0 / 3.0 * height, 7.0/8.0*height))
    # drawer_width =  pyro.sample("drawer_width", dist.Uniform(2.0 / 3.0 * width, 7.0/8.0*width))
    drawer_height = height- thicc
    drawer_width = width - 2*thicc
    drawer_len = 5./6.*length
    drawer_size = make_string((drawer_len, drawer_width, drawer_height))

    drawer_origin = [0.0,0.0,height]
    drawer_quaternion = (1.0, 0.0, 0.0,0.0)
    origin_string = make_string(tuple(drawer_origin))
    quat_string = make_quat_string(drawer_quaternion)

    HANDLE_X = length
    HANDLE_Z = height
    HANDLE_Y = 0

    HANDLE_LEN=pyro.sample("handleLen", dist.Uniform(0.001 / 2, 0.02 / 2)).item()
    HANDLE_WIDTH=pyro.sample("handleWid", dist.Uniform(drawer_width / 6.0 , drawer_width * 3 / 4 )).item()
    HANDLE_HEIGHT=pyro.sample("handleHght", dist.Uniform(height / 12.0 , height / 4.0)).item()
    handle_origin = make_string((HANDLE_X, HANDLE_Y, HANDLE_Z))
    handle_size = make_string((HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT))
    drawer_range = ' "0  ' + str(1.5 * length) + ' " '

    params = np.array([drawer_origin, [side_length, 0.0, 0.0]])

    # ALTERNATE PARAMS
    params[0] = params[0] + [side_length,0.0,height]
    geometry = np.array([length, width, height, left]) # length = 4

    znear, zfar, fovy = get_cam_params()
    obj = ArticulatedObject(1, geometry, params, '', base_xyz, base_quat)

    obj.control = [0,0,0,0,1.5 * length,0,0,0,0,0,0]

    znear_str= make_single_string(znear)
    zfar_str = make_single_string(zfar)
    fovy_str = make_single_string(fovy)

    ## TESTING PARAM CONVERSION HERE
    axis = get_cam_relative_params2(obj)
    # print('get_cam_relative_params2',axis.shape)
    ax_pose = tuple(axis[:3])
    ax_quat = tuple(axis[3:7])
    # print(ax_pose)
    # print(ax_quat)
    # ax_string = make_string(tuple(ax_pose))
    ax_string=make_string(ax_pose)
    # print(ax_string)
    ax_quat_string=make_quat_string(ax_quat)

    xml='''
    <mujoco model="drawer">
        <compiler angle="radian" eulerseq='zxy' />
        <option gravity = "0 0 0" />
        <option>
            <flag contact = "disable"/>
        </option>
        <statistic	extent="1.0" center="0.0 0.0 0.0"/>
        <visual>
            <map fogstart="3" fogend="5" force="0.1" znear='''+znear_str+''' zfar='''+zfar_str+'''/>
        </visual>
        <size njmax="500" nconmax="100" />
        <actuator>
            <velocity joint="drawerJ" name="viva_revolution" kv='10'></velocity>
            <!--position joint="drawerJ" name="viva_position" kp='10'></position-->
        </actuator>
        <asset>
            <texture builtin="flat" name="objtex" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
            <texture builtin="flat" name="handletex" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
            <material name="geomObj" shininess="0.03" specular="0.75" texture="objtex"></material>
            <material name="geomHandle" shininess="0.03" specular="0.75" texture="handletex"></material>
        </asset>
        <worldbody>
            <body name="cabinet_bottom" pos=''' + base_origin + ''' quat='''+base_orientation+'''>
                <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                <geom size='''+ base_size +''' type="box" material="geomObj" name="b"/>
                <body name="cabinet_left" pos=''' + left_origin + '''>
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                    <geom size=''' + side_size + ''' type="box" material="geomObj" name="c" />
                </body>
                <body name="cabinet_right" pos='''+right_origin+'''>
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                    <geom size='''+side_size+''' type="box" material="geomObj" name="d" />
                </body>
                <body name="cabinet_top" pos='''+top_origin+'''>
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                    <geom size='''+top_size+''' type="box" material="geomObj" name="e"/>
                </body>
                <body name="cabinet_back" pos=''' + back_origin + ''' >
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                    <geom size='''+back_size+''' type="box" material="geomObj" name="f" />
                </body>
                <body name="drawer" pos='''+drawer_bottom_origin+''' >
                    <geom size=''' + drawer_bottom_size + ''' type="box" material="geomObj" name="ddddfadfadsfac" />
                    <joint name="drawerJ" pos="0 0 0" axis="1 0 0" type="slide" limited="true" range='''+drawer_range+''' />
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                    <body name="drawer_left" pos=''' + drawer_left_origin + '''>
                        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                        <geom size=''' + drawer_side_size + ''' type="box" material="geomObj" name="ddc" />
                    </body>
                    <body name="drawer_right" pos='''+drawer_right_origin+'''>
                        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                        <geom size='''+drawer_side_size+''' type="box" material="geomObj" name="dddd" />
                    </body>
                    <body name="drawer_front" pos='''+drawer_front_origin+'''>
                        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                        <geom size='''+drawer_back_size+''' type="box" material="geomObj" name="eeeee"/>
                    </body>
                    <body name="drawer_back" pos=''' + drawer_back_origin + ''' >
                        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                        <geom size='''+drawer_back_size+''' type="box" material="geomObj" name="fffff" />
                    </body>
                    <body name="knob" pos='''+handle_origin+'''>
                        <inertial pos="0 0 0" mass="0.1" diaginertia="0.1 0.1 0.1" />
                        <geom size='''+handle_size+''' pos="0 0 0" type="box" material="geomHandle" name="handle"/>
                    </body>
                </body>
            </body>
            <body name="external_camera_body_0" pos="0.0 0 0.00">
                <camera euler="-1.57 1.57 0.0" fovy='''+fovy_str+''' name="external_camera_0" pos="0.0 0 0"></camera>
                <inertial pos= " 0.00 0.0 0.000000 " mass="1" diaginertia="1 1 1" />
                <joint name="cam_j" pos="0.0 0 0" axis = "1 0 0" type="free" />
            </body>
            <!--body name="TESTING" pos='''+ax_string+''' quat='''+ax_quat_string+'''>
                <geom size="0.05" type="sphere" />
            </body-->
        </worldbody>
    </mujoco>'''
    obj.xml = xml
    return obj

def test():
    # import transforms3d as tf3d
    from mujoco_py import load_model_from_xml, MjSim, MjViewer
    from mujoco_py.modder import TextureModder

    for i in range(200):
        l,w,h,t,left,m=sample_drawers()
        drawer = build_drawer(l,w,h,t,left)
        model = load_model_from_xml(drawer.xml)
        sim = MjSim(model)
        viewer = MjViewer(sim)
        modder = TextureModder(sim)
        for name in sim.model.geom_names:
            modder.rand_all(name)

        t = 0
        sim.data.ctrl[0] = 0.05
        while t<1000:
            sim.step()
            viewer.render()
            t += 1

if __name__ == "__main__":
    test()
