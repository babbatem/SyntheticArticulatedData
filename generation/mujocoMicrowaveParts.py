import numpy as np
import pyro
import pyro.distributions as dist

from generation.ArticulatedObjs import Microwave, ArticulatedObject
from generation.utils import sample_quat, sample_pose, make_string, make_single_string, make_quat_string, get_cam_relative_params2, angle_to_quat, get_cam_params

# TODO: fix range given new camera parameters
def sample_microwave():
    # length=pyro.sample("length", dist.Uniform(0.30 / 2, 0.5 / 2)).item()
    # width =pyro.sample('width', dist.Uniform(0.5 / 2, 0.7 / 2)).item()
    # height=pyro.sample('height', dist.Uniform(0.2794 / 2, 0.3048 / 2)).item()

    length=pyro.sample("length", dist.Uniform(10/2*0.0254, 22/2*0.0254)).item()
    width =pyro.sample('width',  dist.Uniform(16/2*0.0254, 30/2*0.0254)).item()
    height=pyro.sample('height', dist.Uniform(9/2*0.0254, 18/2*0.0254)).item()
    thickness=pyro.sample('thicc', dist.Uniform(0.02 / 2, 0.05 / 2)).item()
    # left=pyro.sample('lefty', dist.Bernoulli(0.5)).item()
    left = True
    mass=pyro.sample('mass', dist.Uniform(5.0, 30.0))

    # # ### CABFACTS
    # length = 11.25 * 0.0254 / 2
    # width = 17.75 * 0.0254 / 2
    # height = 9.75 * 0.0254 / 2
    # thickness = 0.02 / 2

    return length, width, height, thickness, left, mass

def sample_handle(side_height):
    HANDLE_LEN=pyro.sample('hl', dist.Uniform(0.02, 0.03)).item()
    HANDLE_WIDTH=pyro.sample('hw', dist.Uniform(0.02, 0.03)).item()
    HANDLE_HEIGHT=pyro.sample('hh', dist.Uniform(0.1, side_height)).item()
    return HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT

def const_handle(side_height):
    HANDLE_LEN = 0.02
    HANDLE_WIDTH = 0.02
    HANDLE_HEIGHT = side_height
    return HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT

def build_microwave(length, width, height, thicc, left, set_pose=None, set_rot=None):

    # TODO: sample shape better
    # TODO: figure out weird door behavior
    # TODO: sample mass and include

    base_length=length
    base_width=width
    base_height=thicc

    if set_pose is None:
        base_xyz, base_angle = sample_pose()
        base_quat = angle_to_quat(base_angle)
        # print('base xyz', base_xyz)
        # print('base angle', base_angle)
        # print('base quat', base_quat)

        # base_quat = sample_quat()
    else:
        # # NOTE: BROKEN
        base_xyz = set_pose
        base_quat = set_rot

    # base_xyz = (2.0, 0.0, 0.0)
    # base_quat = (0.0, 0.0, 0.0, 1.0)

    # base_xyz = (2,0,0)
    # base_quat = (1.0, 0.0,0.0,0.0)

    base_origin=make_string(tuple(base_xyz))
    base_orientation=make_quat_string(tuple(base_quat))

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

    ## I think I need to randomize keypad size...
    # kw_multiplier = pyro.sample("kw", dist.Uniform(1 / 5, 1 / 3)).item()
    kw_multiplier = 1/4
    keypad_size = make_string((side_length - thicc, base_width * kw_multiplier, side_height - thicc))
    keypad_origin = make_string((thicc, base_width - base_width * kw_multiplier -0.001, side_height))

    door_origin=make_string((0.0, base_width * 3 / 4, 0.0))
    hinge_origin=make_string((base_length, -base_width, side_height))
    params = [[base_length, -base_width, side_height], [0.0, base_width, 0.0]]
    hinge_range=' "-2.3 0" '

    # HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT = sample_handle(side_height)
    HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT = const_handle(side_height)
    HANDLE_X = HANDLE_LEN
    HANDLE_Y = 2 * base_width * 3 / 4 - 0.03
    HANDLE_Z = 0

    handle_origin = make_string((HANDLE_X, HANDLE_Y, HANDLE_Z))
    handle_size = make_string((HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT))

    geometry = np.array([length, width, height, left]) # length = 4
    parameters = np.array(params) # shape = 1, 2, 3, length = 6
    znear, zfar, fovy = get_cam_params()
    cab = ArticulatedObject(0, geometry, parameters, '', base_xyz, base_quat)

    znear_str= make_single_string(znear)
    zfar_str = make_single_string(zfar)
    fovy_str = make_single_string(fovy)

    ############################################################################
    ######################         FOR TESTING            ######################
    ############################################################################
    post_params = get_cam_relative_params2(cab)
    # print(post_params)
    axis=post_params[:3]
    axquat=post_params[3:7]
    ax_string = make_string(tuple(axis))
    axquat_string = make_quat_string(axquat)

    # print(ax_string)
    # print(axquat_string)
    ############################################################################
    ############################################################################

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
        <velocity joint="cam_j" name="moving_cam" kv="10" ></velocity>
    </actuator>
    <asset>
        <texture builtin="flat" name="tabletex" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <texture builtin="flat" name="objtex" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <texture builtin="flat" name="handletex" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <texture builtin="flat" name="wallpaper" height="32" width="32" rgb1="1 0 0" type="cube"></texture>
        <texture builtin="flat" name="kp" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <material name="geomTable" shininess="0.03" specular="0.75" texture="tabletex"></material>
        <material name="geomObj" shininess="0.03" specular="0.75" texture="objtex"></material>
        <material name="geomHandle" shininess="0.03" specular="0.75" texture="handletex"></material>
        <material name="ax" shininess="0.03" specular="0.75" texture="wallpaper"></material>
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
                <joint name="bottom_left_hinge" pos="0 0 0" axis="0 0 1" limited="true" range='''+hinge_range+''' />
                <geom size='''+door_size+''' pos='''+door_origin+''' type="box" material="geomObj" name="g"/>
                <body name="handle_link" pos='''+handle_origin+'''>
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                    <geom size='''+handle_size+''' type="box" material="geomHandle" name="h"/>
                </body>
            </body>
        </body>
        <body name="external_camera_body_0" pos="0.0 0 0.00" >
            <camera euler="-1.57 1.57 0.0" fovy='''+fovy+''' name="external_camera_0" pos="0.0 0 0"></camera>
            <inertial pos= " 0.00 0.0 0.000000 " mass="1" diaginertia="1 1 1" />
            <joint name="cam_j" pos="0.0 0 0" axis = "1 0 0" type="slide" />
        </body>
        <!--body name="TESTING" pos='''+ax_string+''' quat='''+axquat_string+'''>
            <geom size="0.05" type="sphere" material="ax"/>
        </body-->
        <!--body> name="x_axis" pos="0.0 0 0.00">
                <geom size="0.1 0.1 0.1" type="box" material="geomHandle" name="who cares1"/>
                <body name="TESTtickles" pos="1 -0.5 -0.5">
                    <geom size="0.1" type="sphere" material="geomTable"/>
                </body>
                <body name="TESTYdf" pos="1 -0.5 0.5">
                    <geom size="0.1" type="sphere" material="geomTable"/>
                </body>
                <body name="TESTYafd" pos="1 0.5 0.5">
                    <geom size="0.1" type="sphere" material="geomTable"/>
                </body>
                <body name="TESTYaaaa" pos="1 0.5 -0.5">
                    <geom size="0.1" type="sphere" material="geomTable"/>
                </body>
                <body name="TESTtickles2" pos="2 -0.5 -0.5">
                    <geom size="0.1" type="sphere" material="geomTable"/>
                </body>
                <body name="TESTYdf2" pos="2 -0.5 0.5">
                    <geom size="0.1" type="sphere" material="geomTable" />
                </body>
                <body name="TESTYafd2" pos="2 0.5 0.5">
                    <geom size="0.1" type="sphere" material="geomTable"/>
                </body>
                <body name="TESTYaaaa2" pos="2 0.5 -0.5">
                    <geom size="0.1" type="sphere" material="geomTable"/>
                </body>
        </body-->
        <!--body> name="y_axis" pos="0.0 0 0.00">
                <geom size="0.01 10 0.01" type="box" material="geomHandle" name="who cares2"/>
        </body>
        <body> name="z_axis" pos="0.0 0 0.00">
                <geom size="0.01 0.01 10" type="box" material="geomHandle" name="who cares3"/>
        </body>
        <body> name="testx" pos="1.0 0 0.00">
                <geom size="0.1 0.1 0.1" type="box" material="geomHandle"/>
        </body>
        <body> name="testy" pos="1.0 1.0 0.00">
                <geom size="0.1 0.1 0.1" type="box" material="geomHandle"/>
        </body>
        <body> name="testz" pos="1.0 0 1.0">
                <geom size="0.1 0.1 0.1" type="box" material="geomHandle"/>
        </body-->
    </worldbody>
</mujoco>'''

def test():
    # import matplotlib
    # matplotlib.use('Agg')
    # from matplotlib import pyplot as plt

    l,w,h,t,left,m=sample_microwave()
    # cab=build_microwave(l,w,h,t,left,set_pose=[1.2,0.0,-0.7], set_rot = [0,0,0,1])
    cab=build_microwave(l,w,h,t,left)
    # print(cab.xml)
    model = load_model_from_xml(cab.xml)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    modder = TextureModder(sim)
    for name in sim.model.geom_names:
        modder.rand_all(name)

    t = 0
    sim.data.ctrl[0] = - 0.2
    # print(sim.model.stat)
    while t < 1000:
        sim.step()
        viewer.render()
        # print(sim.data.qpos)
        # img, depth = sim.render(192, 108, camera_name='external_camera_0', depth=True)
        # plt.subplot(121); plt.imshow(img);
        # plt.subplot(122); plt.imshow(depth);
        # plt.savefig('microtest.png')
        t += 1

if __name__ == "__main__":
    from mujoco_py import load_model_from_xml, MjSim, MjViewer
    from mujoco_py.modder import TextureModder
    for i in range(200):
        test()
