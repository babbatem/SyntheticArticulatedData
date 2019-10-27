import numpy as np
import pyro
import pyro.distributions as dist
import torch
import transforms3d as tf3d

from magic.data.generation.ArticulatedObjs import ArticulatedObject
from magic.data.generation.utils import sample_quat, sample_pose, make_string, make_single_string, make_quat_string, get_cam_relative_params2, angle_to_quat, get_cam_params, transform_param

d_len = dist.Uniform(0.28,0.32)
d_width = dist.Uniform(0.3,0.7)
d_height = dist.Uniform(0.5,0.7)
d_thicc = dist.Uniform(0.03, 0.05)
d_mass = dist.Uniform(5.0, 30.0)

def sample_cabinet2(mean_flag):
    if mean_flag:
        length=d_len.mean
        width=d_width.mean
        height=d_height.mean
        thickness=d_thicc.mean
        mass=d_mass.mean
    else:
        length=pyro.sample("length", d_len).item()
        width =pyro.sample('width', d_width).item()
        height=pyro.sample('height', d_height).item()
        thickness=pyro.sample('thicc', d_thicc).item()
        mass=pyro.sample('mass', d_mass)
    left = True
    return length / 2, width / 2, height / 2, thickness / 2, left, mass

def sample_handle(length, width, height, left):
    HANDLE_LEN=pyro.sample('hl', dist.Uniform(0.01, 0.03)).item()
    HANDLE_WIDTH=pyro.sample('hw', dist.Uniform(0.01, 0.03)).item()
    HANDLE_HEIGHT=pyro.sample('hh', dist.Uniform(0.01, 0.2)).item()

    HX = length/4
    HY = pyro.sample('hy', dist.Uniform(width*2 - width/4, 2 * width - 2*HANDLE_WIDTH)) / 2
    HZ = pyro.sample('hz', dist.Uniform(-(height - HANDLE_HEIGHT), height-HANDLE_HEIGHT))
    if left:
        HY = HY
    else:
        HY = -HY

    return HX, HY, HZ, HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT

def build_cabinet2(length, width, height, thicc, left, set_pose=None, set_rot=None):

    base_length=length
    base_width=width
    base_height=thicc

    if not set_pose:
        base_xyz, base_angle = sample_pose()
        base_quat = angle_to_quat(base_angle)
    else:
        base_xyz = tuple(set_pose)
        base_quat = tuple(set_rot)

    base_origin=make_string(base_xyz)
    base_orientation=make_quat_string(base_quat)

    base_size = make_string((base_length, base_width, base_height))
    side_length=length
    side_width=thicc
    side_height=height
    side_size = make_string((side_length, side_width, side_height))

    back_size = make_string((side_width, base_width, side_height))
    top_size = base_size
    door_size = make_string((side_width, base_width / 2, side_height))

    left_origin  = make_string((0, -width + thicc, height))
    right_origin = make_string((0, width - thicc, height))
    top_origin = make_string((0,0,height*2))
    back_origin = make_string((-base_length + thicc, 0.0, height))

    HANDLE_X, HANDLE_Y, HANDLE_Z, HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT = sample_handle(length, width, height, 1)
    handle_size = make_string((HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT))

    param_axis1 = [base_length, -base_width, side_height]
    param_radius1 = [0.0, base_width, 0.0]
    door1_origin=make_string((0.0, base_width / 2, 0.0))
    hinge1_origin=make_string(tuple(param_axis1))
    hinge1_range=' "-2.3 0" '
    handle1_origin = make_string((HANDLE_X, HANDLE_Y, HANDLE_Z))

    param_axis2 = [base_length, base_width, side_height]
    param_radius2 = [0.0, -base_width, 0.0]
    door2_origin=make_string((0.0, - base_width / 2, 0.0))
    hinge2_origin=make_string(tuple(param_axis2))
    hinge2_range=' "0 2.3" '
    handle2_origin = make_string((HANDLE_X, -HANDLE_Y, HANDLE_Z))

    znear, zfar, fovy = get_cam_params()
    znear_str= make_single_string(znear)
    zfar_str = make_single_string(zfar)
    fovy_str = make_single_string(fovy)
    geometry = np.array([length, width, height, left]) # length = 4

    parameters = np.array([[param_axis1, param_radius1],[param_axis2, param_radius2] ]) # shape = 1, 2, 3, length = 6
    cab = ArticulatedObject(4, geometry, parameters, '', base_xyz, base_quat)

    # FOR TESTING
    ax = cab.params[0][0]
    d = cab.params[0][1]

    post_params, door_param = transform_param(ax, d, cab)
    post_params2,door_param2 = transform_param(cab.params[1][0],cab.params[1][1],cab)

    axis=post_params[:3]
    axquat=post_params[3:7]

    ax_x_string = make_string(tuple(axis))
    axquat_string = make_quat_string(axquat)

    axis2=post_params2[:3]
    axquat2=post_params2[3:7]
    ax_x_string2 = make_string(tuple(axis2))
    axquat_string2 = make_quat_string(axquat2)
    xml='''
<mujoco model="cabinet">
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
        <velocity joint="bottom_left_hinge" name="viva_revolution" kv='10'></velocity>
        <velocity joint="bottom_right_hinge" name="viva" kv='10'></velocity>
        <!--position joint="bottom_left_hinge" name="viva_positionL" kp='10'></position-->
        <!--position joint="bottom_right_hinge" name="viva_positionR" kp='10'></position-->
    </actuator>
    <asset>
        <texture builtin="flat" name="tabletex" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <texture builtin="flat" name="objtex" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <texture builtin="flat" name="handletex" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <texture builtin="flat" name="wallpaper" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <material name="geomTable" shininess="0.03" specular="0.75" texture="tabletex"></material>
        <material name="geomObj" shininess="0.03" specular="0.75" texture="objtex"></material>
        <material name="geomHandle" shininess="0.03" specular="0.75" texture="handletex"></material>
        <material name="bg" shininess="0.03" specular="0.75" texture="wallpaper"></material>
    </asset>
    <worldbody>
            <!--body name="obj_x_axis" pos='''+ax_x_string+''' quat='''+axquat_string+'''>
                    <geom size="0.1" type="sphere" material="geomHandle" />
            </body>
            <body name="obj_2_axis" pos='''+ax_x_string2+''' quat='''+axquat_string2+'''>
                    <geom size="0.1" type="sphere" material="geomHandle" />
            </body-->
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
                <body name="cabinet_left_hinge" pos='''+hinge1_origin+'''>
                    <inertial pos='''+door1_origin+''' mass="1" diaginertia="1 1 1" />
                    <joint name="bottom_left_hinge" pos="0 0 0" axis="0 0 1" limited="true" range='''+hinge1_range+''' />
                    <geom size='''+door_size+''' pos='''+door1_origin+''' type="box" material="geomObj" name="g"/>
                    <body name="handle_link" pos='''+handle1_origin+'''>
                        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                        <geom size='''+handle_size+''' type="box" material="geomHandle" name="h"/>
                    </body>
                </body>
                <body name="cabinet_right_hinge" pos='''+hinge2_origin+'''>
                    <inertial pos='''+door2_origin+''' mass="1" diaginertia="1 1 1" />
                    <joint name="bottom_right_hinge" pos="0 0 0" axis="0 0 1" limited="true" range='''+hinge2_range+''' />
                    <geom size='''+door_size+''' pos='''+door2_origin+''' type="box" material="geomObj" name="q"/>
                    <body name="handle2_link" pos='''+handle2_origin+'''>
                        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                        <geom size='''+handle_size+''' type="box" material="geomHandle" name="q2"/>
                    </body>
                </body>


            </body>
        <body name="external_camera_body_0" pos="0.0 0 0.00">
            <camera euler="-1.57 1.57 0.0" fovy='''+fovy_str+''' name="external_camera_0" pos="0.0 0 0"></camera>
            <inertial pos= " 0.00 0.0 0.000000 " mass="1" diaginertia="1 1 1" />
            <joint name="cam_j" pos="0.0 0 0" axis = "1 0 0" type="free" />
        </body>
    </worldbody>
</mujoco>'''

    # geometry = np.array([length, width, height, left]) # length = 4
    # parameters = np.array(params) # shape = 1, 2, 3, length = 6
    # cab = Cabinet(0, geometry, parameters, xml, pose=base_xyz, rotation=base_angle)
    cab.xml=xml
    return cab

def set_two_door_control(sim, obj):
    max_vel = 0.3
    mode = np.random.choice([0,1,2,3])
    open_right = np.random.binomial(1,0.75)
    open_left = np.random.binomial(1,0.75)
    right_mult = np.random.rand()
    left_mult = np.random.rand()
    maxxx = max(right_mult, left_mult)

    # mode = 3
    # open_right = False

    if mode == 0: # start with both doors closed

        if open_right:
            if open_left:
                sim.data.ctrl[1] = max_vel * right_mult / maxxx
                sim.data.ctrl[0] = -max_vel * left_mult / maxxx if obj=='cabinet2' else max_vel * left_mult / maxxx
            else:
                sim.data.ctrl[1] = max_vel
        else:
            sim.data.ctrl[0] = -max_vel if obj=='cabinet2' else max_vel

    elif mode == 1:
        # start with left open
        sim.data.ctrl[0] = -2.0 if obj=='cabinet2' else 2.0
        for i in range(1000):
            sim.step()
            # viewer.render()

        if open_right:
            if open_left:
                sim.data.ctrl[1] = max_vel * right_mult / maxxx
                sim.data.ctrl[0] = max_vel * left_mult / maxxx if obj=='cabinet2' else -max_vel * left_mult / maxxx
            else:
                sim.data.ctrl[1] = max_vel
        else:
            sim.data.ctrl[0] = max_vel if obj=='cabinet2' else -max_vel

    elif mode == 2:
        sim.data.ctrl[1] = 2.0
        for i in range(1000):
            sim.step()

        if open_right:
            if open_left:
                sim.data.ctrl[1] = -max_vel * right_mult / maxxx
                sim.data.ctrl[0] = -max_vel * left_mult / maxxx if obj=='cabinet2' else max_vel * left_mult / maxxx
            else:
                sim.data.ctrl[1] = -max_vel
        else:
            sim.data.ctrl[0] = -max_vel if obj=='cabinet2' else max_vel



    else:
        # start with both open
        sim.data.ctrl[1] = 2.0
        sim.data.ctrl[0] = -2.0 if obj=='cabinet2' else 2.0
        for i in range(1000):
            sim.step()
            # viewer.render()

        if open_right:
            if open_left:
                sim.data.ctrl[1] = -max_vel * right_mult / maxxx
                sim.data.ctrl[0] = max_vel * left_mult / maxxx if obj=='cabinet2' else -max_vel * left_mult / maxxx
            else:
                sim.data.ctrl[1] = -max_vel
        else:
            sim.data.ctrl[0] = max_vel if obj=='cabinet2' else -max_vel


def test(k=0):
    from mujoco_py import load_model_from_xml, MjSim, MjViewer
    from mujoco_py.modder import TextureModder

    l,w,h,t,left,m=sample_cabinet2()
    cab=build_cabinet2(l,w,h,t,left)
    # print(cab.xml)
    model = load_model_from_xml(cab.xml)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    modder = TextureModder(sim)
    for name in sim.model.geom_names:
        modder.rand_all(name)

    set_two_door_control(sim)
    q1s=[]
    q2s=[]
    t = 0
    # mode 0 indicates starting lc
    while t < 4000:
        # for name in sim.model.geom_names:
        #     modder.rand_all(name)
        viewer.render()
        if t % 250 == 0:
            q1 = sim.data.qpos[0]
            q2 = sim.data.qpos[1]
            print(sim.data.qpos)
            q1s.append(q1)
            q2s.append(q2)

        sim.step()
        t += 1
    # print(q1s)
    np.save('devdata/q1_'+str(k).zfill(3), q1s)
    np.save('devdata/q2_'+str(k).zfill(3), q2s)

if __name__ == '__main__':
    for i in range(40):
        test(k=i)
