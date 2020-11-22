import numpy as np
import pyro
import pyro.distributions as dist
import torch
import transforms3d as tf3d

from generation.ArticulatedObjs import ArticulatedObject
from generation.utils import sample_quat, sample_pose, make_string, make_single_string, make_quat_string, get_cam_relative_params2, angle_to_quat, get_cam_params

d_len = dist.Uniform(8/2*0.0254, 20/2*0.0254)
d_width = dist.Uniform(16/2*0.0254, 24/2*0.0254)
d_height = dist.Uniform(8/2*0.0254, 20/2*0.0254)
d_thicc = dist.Uniform(0.02 / 2, 0.05 / 2)
d_mass = dist.Uniform(5.0, 30.0)

def sample_toaster(mean_flag):
    if mean_flag:
        length = d_len.mean
        width = d_width.mean
        height = d_height.mean
        thickness=d_thicc.mean
        mass = d_mass.mean
    else:
        length=pyro.sample("length",d_len).item()
        width =pyro.sample('width',d_width).item()
        height=pyro.sample('height',d_height).item()
        thickness=pyro.sample('thicc',d_thicc).item()
        mass = pyro.sample('mass', d_mass).item()
    left = False
    return length, width, height, thickness, left, mass

def sample_t_handle(side_width):
    HANDLE_LEN=pyro.sample('hl', dist.Uniform(0.01, 0.04)).item()
    HANDLE_WIDTH=pyro.sample('hw', dist.Uniform(side_width, side_width)).item()
    HANDLE_HEIGHT=pyro.sample('hh', dist.Uniform(0.01, 0.04)).item()
    return HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT

def build_toaster(length, width, height, thicc, left, set_pose=None, set_rot=None):
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

    left_origin  = make_string((0, -width + thicc, height))
    right_origin = make_string((0, width - thicc, height))
    top_origin = make_string((0,0,height*2))
    back_origin = make_string((-base_length + thicc, 0.0, height))

    ## I think I need to randomize keypad size...
    kw_multiplier = pyro.sample("kw", dist.Uniform(1 / 5, 1 / 3)).item()
    keypad_size = make_string((side_length - thicc, base_width * kw_multiplier, side_height - thicc))
    keypad_origin = make_string((thicc, base_width - base_width * kw_multiplier -0.001, side_height))

    door_origin=make_string((0.0+0.02, 0, side_height))
    door_size = make_string((side_width, base_width * (1-kw_multiplier), side_height))

    hinge_origin=make_string((base_length, -base_width * kw_multiplier, 0))
    params = [[base_length, -base_width * (kw_multiplier), 0], [0.0, 0.0, side_height]]
    hinge_range=' "0 2.3" '

    HANDLE_X = length/4
    HANDLE_Y = 0
    HANDLE_Z = 2*side_height - 2*thicc
    HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT = sample_t_handle(base_width * kw_multiplier)
    handle_origin = make_string((HANDLE_X, HANDLE_Y, HANDLE_Z))
    handle_size = make_string((HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT))

    geometry = np.array([-length, width, height, left]) # length = 4
    parameters = np.array(params) # shape = 1, 2, 3, length = 6


    znear, zfar, fovy = get_cam_params()

    cab = ArticulatedObject(3, geometry, parameters, '', base_xyz, base_quat)

    cab.control = [2,2,2,2,2,2,2,2,2]

    znear_str= make_single_string(znear)
    zfar_str = make_single_string(zfar)
    fovy_str = make_single_string(fovy)

    post_params = get_cam_relative_params2(cab)
    # print(post_params)
    axis=post_params[:3]
    axquat=post_params[3:7]
    ax_string = make_string(tuple(axis))
    axquat_string = make_quat_string(axquat)

    xml=write_xml(ax_string, axquat_string, base_origin, base_orientation, base_size, \
                    left_origin, right_origin, side_size, \
                    top_origin, top_size, \
                    back_origin, back_size, \
                    keypad_origin, keypad_size, \
                    hinge_origin, hinge_range, \
                    door_origin, door_size, handle_origin, handle_size, \
                    znear_str, zfar_str, fovy_str)

    # geometry = np.array([length, width, height, left]) # length = 4
    # parameters = np.array(params) # shape = 1, 2, 3, length = 6
    # cab = Cabinet(0, geometry, parameters, xml, pose=base_xyz, rotation=base_angle)
    cab.xml=xml
    return cab

def write_xml(ax_string, axquat_string, base_origin, base_orientation, base_size, \
                left_origin, right_origin, side_size, \
                top_origin, top_size, \
                back_origin, back_size, \
                keypad_origin, keypad_size, \
                hinge_origin, hinge_range, \
                door_origin, door_size, handle_origin, handle_size, \
                znear, zfar, fovy):
    return '''
<mujoco model="cabinet">
    <compiler angle="radian" eulerseq='zxy' />
    <option gravity = "0 0 0" />
    <option>
        <flag contact = "disable"/>
    </option>
    <statistic	extent="1.0" center="0.0 0.0 0.0"/>
    <visual>
        <map fogstart="3" fogend="5" force="0.1" znear='''+znear+''' zfar='''+zfar+'''/>
    </visual>
    <size njmax="500" nconmax="100" />
    <actuator>
        <velocity joint="bottom_left_hinge" name="viva_revolution" kv='10'></velocity>
        <!--position joint="bottom_left_hinge" name="viva_position" kp='10'></position-->
    </actuator>
    <asset>
        <texture builtin="flat" name="tabletex" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <texture builtin="flat" name="objtex" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <texture builtin="flat" name="handletex" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <texture builtin="flat" name="wallpaper" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <texture builtin="flat" name="kp" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <material name="geomTable" shininess="0.03" specular="0.75" texture="tabletex"></material>
        <material name="geomObj" shininess="0.03" specular="0.75" texture="objtex"></material>
        <material name="geomHandle" shininess="0.03" specular="0.75" texture="handletex"></material>
        <material name="bg" shininess="0.03" specular="0.75" texture="wallpaper"></material>
        <material name="geomKeypad" shininess="0.03" specular="0.75" texture="kp"></material>
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
            <body name="cabinet_keypad" pos='''+keypad_origin+'''>
                <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                <geom size='''+keypad_size+''' type="box" material="geomKeypad" name="keypizzlemynizzle" />
            </body>
            <body name="cabinet_left_hinge" pos='''+hinge_origin+'''>
                <inertial pos='''+door_origin+''' mass="1" diaginertia="1 1 1" />
                <joint name="bottom_left_hinge" type="hinge" pos="0 0 0" axis="0 1 0" limited="true" range='''+hinge_range+''' />
                <geom size='''+door_size+''' pos='''+door_origin+''' type="box" material="geomObj" name="g"/>
                <body name="handle_link" pos='''+handle_origin+'''>
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                    <geom size='''+handle_size+''' type="box" material="geomHandle" name="h"/>
                </body>
            </body>
        </body>
        <body name="external_camera_body_0" pos="0.0 0 0.00">
            <camera euler="-1.57 1.57 0.0" fovy='''+fovy+''' name="external_camera_0" pos="0.0 0 0"></camera>
            <inertial pos= " 0.00 0.0 0.000000 " mass="1" diaginertia="1 1 1" />
            <joint name="cam_j" pos="0.0 0 0" axis = "1 0 0" type="free" />
        </body>
        <!--body> name="x_axis" pos="0.0 0 0.00">
                <geom size="10 0.01 0.01" type="box" material="geomHandle" name="who cares1"/>
                <body name="TESTING" pos='''+ax_string+''' quat='''+axquat_string+'''>
                    <geom size="0.1" type="sphere"/>
                </body>
        </body>
        <body> name="y_axis" pos="0.0 0 0.00">
                <geom size="0.01 10 0.01" type="box" material="geomHandle" name="who cares2"/>
        </body>
        <body> name="z_axis" pos="0.0 0 0.00">
                <geom size="0.01 0.01 10" type="box" material="geomHandle" name="who cares3"/>
        </body-->
    </worldbody>
</mujoco>'''

def test():
    from mujoco_py import load_model_from_xml, MjSim, MjViewer
    from mujoco_py.modder import TextureModder

    l,w,h,t,left,m=sample_toaster()
    cab=build_toaster(l,w,h,t,left)
    # print(cab.xml)
    model = load_model_from_xml(cab.xml)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    modder = TextureModder(sim)
    for name in sim.model.geom_names:
        modder.rand_all(name)

    t = 0
    sim.data.ctrl[0] = 0.2
    while t < 1000:
        # for name in sim.model.geom_names:
        #     modder.rand_all(name)
        sim.step()
        viewer.render()
        t += 1

if __name__ == "__main__":
    for i in range(200):
        test()
