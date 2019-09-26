import numpy as np
import pyro
import pyro.distributions as dist

from generation.ArticulatedObjs import ArticulatedObject
from generation.utils import *

# TODO: how to sample geometry more realistically?
# ie, what's the joint distribution over length, width, height
def sample_refrigerator():
    length = pyro.sample("length", dist.Uniform(24/2*0.0254, 33/2*0.0254)).item()
    width =pyro.sample('width',  dist.Uniform(23/2*0.0254, 36/2*0.0254)).item()
    height=pyro.sample('height', dist.Uniform(65/2*0.0254, 69/2*0.0254)).item()
    thickness=pyro.sample('thicc', dist.Uniform(0.02 / 2, 0.05 / 2)).item()
    mass=pyro.sample('mass', dist.Uniform(5.0, 30.0))
    left=True
    return length, width, height, thickness, left, mass

def sample_fridge_handle(length, width, height, left):
    HANDLE_LEN=pyro.sample('hl', dist.Uniform(0.01, 0.03)).item()
    HANDLE_WIDTH=pyro.sample('hw', dist.Uniform(0.01, 0.05)).item()
    HANDLE_HEIGHT=pyro.sample('hh', dist.Uniform(height / 4, height)).item()

    HX = HANDLE_LEN
    HY = -width * 2 + HANDLE_WIDTH
    HZ = pyro.sample('hz', dist.Uniform(-(height - HANDLE_HEIGHT), height-HANDLE_HEIGHT))
    return HX, HY, HZ, HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT

def build_refrigerator(length, width, height, thicc, left, set_pose=None, set_rot=3.14):

    base_length=length
    base_width=width
    base_height=thicc

    if set_pose is None:
        base_xyz, base_angle = sample_pose_fridge(base_length, base_width)
        base_quat = angle_to_quat(base_angle)
        # print('base xyz', base_xyz)
        # print('base angle', base_angle)
        # print('base quat', base_quat)

        # base_quat = sample_quat()
    else:
        # # NOTE: BROKEN
        base_xyz = set_pose
        base_angle = set_rot
        base_quat = angle_to_quat(base_angle)

    # # FOR TESTING ############################
    # base_xyz = (2.0, 0.0, 0.0)
    # base_quat = (0.0, 0.0, 0.0, 1.0)
    # ########################################################

    # build the case
    base_origin=make_string(base_xyz)
    base_orientation=make_quat_string(angle_to_quat(base_angle))

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

    # sample a height fraction; maybe move to sample_refrigerator, but sounds annoying.
    height_fraction = pyro.sample("hf", dist.Uniform(10. / 16., 13. / 16. )).item()

    # build the doors
    HANDLE_X, HANDLE_Y, HANDLE_Z, HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT = sample_fridge_handle(length, width, height * height_fraction, 1)
    handle_size = make_string((HANDLE_LEN, HANDLE_WIDTH, HANDLE_HEIGHT))
    param_axis1 = [base_length, base_width, side_height * (height_fraction)]
    param_radius1 = [0.0, base_width, 0.0]
    door1_origin=make_string((0.0, -base_width, 0.0))
    door_size1 = make_string((side_width, base_width, side_height * height_fraction))
    hinge1_origin=make_string(tuple(param_axis1))
    hinge1_range=' "0 2.3" '
    handle1_origin = make_string((HANDLE_X, HANDLE_Y, HANDLE_Z))

    HANDLE_X2, HANDLE_Y2, HANDLE_Z2, HANDLE_LEN2, HANDLE_WIDTH2, HANDLE_HEIGHT2 = sample_fridge_handle(length, width, height * (1-height_fraction), 1)
    handle_size2 = make_string((HANDLE_LEN2, HANDLE_WIDTH2, HANDLE_HEIGHT2))
    param_axis2 = [base_length, base_width, 2*side_height * (height_fraction) + side_height*(1-height_fraction)]
    param_radius2 = [0.0, base_width, 0.0]
    door2_origin=make_string((0.0, -base_width, 0.0))
    door_size2 = make_string((side_width, base_width, side_height * (1-height_fraction)))
    hinge2_origin=make_string(tuple(param_axis2))
    hinge2_range=' "0 2.3" '
    handle2_origin = make_string((HANDLE_X2, HANDLE_Y2, HANDLE_Z2))

    # build the shelf
    shelf_size = base_size
    shelf_origin = make_string((0,0,height*2*height_fraction))

    # camera params
    znear, zfar, fovy = get_cam_params()
    znear_str= make_single_string(znear)
    zfar_str = make_single_string(zfar)
    fovy_str = make_single_string(fovy)

    # record labels
    geometry = np.array([length, width, height, left]) # length = 4
    parameters = np.array([[param_axis1, param_radius1],[param_axis2, param_radius2] ]) # shape = 1, 2, 3, length = 6

    # construct the object
    fridge = ArticulatedObject(4, geometry, parameters, '', base_xyz, base_quat)

    # FOR TESTING. compute the axis poses
    ax = fridge.params[0][0]
    d = fridge.params[0][1]
    post_params, door_param = transform_param(ax, d, fridge)
    post_params2,door_param2 = transform_param(fridge.params[1][0],fridge.params[1][1],fridge)
    axis=post_params[:3]
    axquat=post_params[3:7]
    ax_x_string = make_string(tuple(axis))
    axquat_string = make_quat_string(axquat)
    axis2=post_params2[:3]
    axquat2=post_params2[3:7]
    ax_x_string2 = make_string(tuple(axis2))
    axquat_string2 = make_quat_string(axquat2)

    # fill in the blanks to generate the xml for the object
    xml='''
<mujoco model="fridgeinet">
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
            <body name="fridgeinet_bottom" pos=''' + base_origin + ''' quat='''+base_orientation+'''>
                <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                <geom size='''+ base_size +''' type="box" material="geomObj" name="b"/>
                <body name="fridgeinet_left" pos=''' + left_origin + '''>
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                    <geom size=''' + side_size + ''' type="box" material="geomObj" name="c" />
                </body>
                <body name="fridgeinet_right" pos='''+right_origin+'''>
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                    <geom size='''+side_size+''' type="box" material="geomObj" name="d" />
                </body>
                <body name="fridgeinet_top" pos='''+top_origin+'''>
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                    <geom size='''+top_size+''' type="box" material="geomObj" name="e"/>
                </body>
                <body name="fridgeinet_shelf" pos='''+shelf_origin+'''>
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                    <geom size='''+shelf_size+''' type="box" material="geomObj" name="shelf"/>
                </body>
                <body name="fridgeinet_back" pos=''' + back_origin + ''' >
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                    <geom size='''+back_size+''' type="box" material="geomObj" name="f" />
                </body>
                <body name="fridgeinet_left_hinge" pos='''+hinge1_origin+'''>
                    <inertial pos='''+door1_origin+''' mass="1" diaginertia="1 1 1" />
                    <joint name="bottom_left_hinge" pos="0 0 0" axis="0 0 1" limited="true" range='''+hinge1_range+''' />
                    <geom size='''+door_size1+''' pos='''+door1_origin+''' type="box" material="geomObj" name="g"/>
                    <body name="handle_link" pos='''+handle1_origin+'''>
                        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                        <geom size='''+handle_size+''' type="box" material="geomHandle" name="h"/>
                    </body>
                </body>
                <body name="fridgeinet_right_hinge" pos='''+hinge2_origin+'''>
                    <inertial pos='''+door2_origin+''' mass="1" diaginertia="1 1 1" />
                    <joint name="bottom_right_hinge" pos="0 0 0" axis="0 0 1" limited="true" range='''+hinge2_range+''' />
                    <geom size='''+door_size2+''' pos='''+door2_origin+''' type="box" material="geomObj" name="q"/>
                    <body name="handle2_link" pos='''+handle2_origin+'''>
                        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                        <geom size='''+handle_size2+''' type="box" material="geomHandle" name="q2"/>
                    </body>
                </body>
            </body>
        <body name="external_camera_body_0" pos="0.0 0 0.00">
            <camera euler="-1.57 1.57 0.0" fovy='''+fovy_str+''' name="external_camera_0" pos="0.0 0 0"></camera>
        </body>
    </worldbody>
</mujoco>'''
    fridge.xml=xml
    return fridge

def test():
    import cv2
    from mujoco_py import load_model_from_xml, MjSim, MjViewer
    from mujoco_py.modder import TextureModder
    l,w,h,t,left,m=sample_refrigerator()
    fridge=build_refrigerator(l,w,h,t,left,
                              set_pose = (3.0, 0.0, -1.0))

    model = load_model_from_xml(fridge.xml)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    modder = TextureModder(sim)

    t = 0
    sim.data.ctrl[0] =  0.1
    sim.data.ctrl[1] =  0.2
    while t < 2000:
        sim.step()
        if t % 100 == 0:
            img, depth = sim.render(256, 212, camera_name='external_camera_0', depth=True)
            img=np.flip(img, axis=0)
            cv2.imwrite('refrigerator' + str(t) +  '.png', img)
        viewer.render()
        t += 1

if __name__ == "__main__":
    # for i in range(200):
    test()
