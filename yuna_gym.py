"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Joint Monkey
------------
- Animates degree-of-freedom ranges for a given asset.
- Demonstrates usage of DOF properties and states.
- Demonstrates line drawing utilities to visualize DOF frames (origin and axis).
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil
import time
from CPG.calculate_limitcycle import *
from CPG.updateCPGStance import *
from CPG.CPG_controller import *

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# simple asset descriptor for selecting from a list

class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments

asset_descriptors = [
    AssetDesc("mjcf/nv_humanoid.xml", False),
    AssetDesc("mjcf/nv_ant.xml", False),
    AssetDesc("urdf/cartpole.urdf", False),
    AssetDesc("urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf", False),
    AssetDesc("urdf/franka_description/robots/franka_panda.urdf", True),
    AssetDesc("urdf/kinova_description/urdf/kinova.urdf", False),
    AssetDesc("urdf/anymal_b_simple_description/urdf/anymal.urdf", True),
    AssetDesc("urdf/yuna/urdf/yuna.urdf", False),
]

# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {"name": "--asset_id", "type": int, "default": 7, "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)},
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
    quit()

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_root = "../../assets"
asset_file = asset_descriptors[args.asset_id].file_name

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.array([0, 0, -1.57,
                     0, 0, -1.57,
                     0, 0, -1.57,
                     0, 0, 1.57,
                     0, 0, 1.57,
                     0, 0, 1.57], dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states['pos']

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        # make sure our default position is in range
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]
    else:
        # set reasonable animation limits for unlimited joints
        if dof_types[i] == gymapi.DOF_ROTATION:
            # unlimited revolute joint
            lower_limits[i] = -math.pi
            upper_limits[i] = math.pi
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            # unlimited prismatic joint
            lower_limits[i] = -1.0
            upper_limits[i] = 1.0
    # set DOF position to default
    dof_positions[i] = defaults[i]
    # set speed depending on DOF type and range of motion
    if dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

# Print DOF properties
for i in range(num_dofs):
    print("DOF %d" % i)
    print("  Name:     '%s'" % dof_names[i])
    print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
    print("  Stiffness:  %r" % stiffnesses[i])
    print("  Damping:  %r" % dampings[i])
    print("  Armature:  %r" % armatures[i])
    print("  Limited?  %r" % has_limits[i])
    if has_limits[i]:
        print("    Lower   %f" % lower_limits[i])
        print("    Upper   %f" % upper_limits[i])

# set up the env grid
num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(10, 5, 10)
cam_target = gymapi.Vec3(3, -2.5, 8)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

from setup.xMonsterKinematics import *
xmk = HexapodKinematics()

# CPG
T = 5000 #Time in seconds code operates for
nIter = int(round(T/0.01))
pi = math.pi
#creating cpg dict
cpg = {
    'initLength': 0,
    's': 0.15 * np.ones(6),   # stride length
    'nomX': np.array([0.51589,  0.51589,  0.0575,   0.0575, - 0.45839, - 0.45839]),
    'nomY': np.array([0.23145, - 0.23145,   0.5125, - 0.5125,   0.33105, - 0.33105]),
    'h': 0.2249,
    's1OffsetY': np.array([0.2375*math.sin(pi/6), -0.2375*math.sin(pi/6), 0.1875, -0.1875, 0.2375*math.sin(pi/6), -0.2375*math.sin(pi/6)]),#robot measurement;  distance on y axis from robot center to base_actuator
    's1OffsetAngY': np.array([-pi/3, pi/3, 0, 0, pi/3, -pi/3]),
    'n': 2,  #limit cycle shape 2:standard, 4:super
    'b': np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6]), #np.array([.4, .4, .4, .4, .4, .4]), #TUNEABLE: step height in radians %1.0
    'scaling': 10, #TUNEABLE: shifts the units into a reasonable range for cpg processing (to avoid numerical issues)
    'shouldersCorr': np.array([-1, 1, -1, 1, -1, 1]),
    'phase_lags': np.array([pi, pi, 0, pi, 0, pi]),
    'dynOffset': np.zeros([3,6]), #Offset on joints developed through constraining
    'dynOffsetInc': np.zeros([3,6]), #Increment since last iteration
    'x': np.zeros([nIter,6]), #TUNEABLE: Initial CPG x-positions
    'y': np.zeros([nIter,6]), #TUNEABLE: Initial CPG y-positions
    'x0': np.zeros([1,6]), #limit cycle center x
    'y0': np.zeros([6,6]), #limit cycle center y
    'legs': np.zeros([1,18]), #Joint angle values
    'elbowsLast': np.zeros([1,6]), #elbow values
    'torques': np.zeros([1,18]), #Joint torque values
    'torqueOffsets': np.zeros([1,18]), #Joint torque offset values
    'gravCompTorques': np.zeros([1,18]), #Joint torque values
    'forces': np.zeros([3,6]), #ee force values
    'gravCompForces': np.zeros([3,6]), #Joint torque values
    'forceStance': np.zeros([1,6]), #grounded legs determined by force
    'CPGStance': np.array([False,False,False,False,False,False]), #grounded legs determined by position (lower tripod)
    'CPGStanceDelta': np.zeros([1,6]), #grounded legs determined by position (lower tripod)
    'CPGStanceBiased': np.zeros([1,6]), #grounded legs determined by position (lower tripod)
    'comm_alpha': 1.0, #commanded alpha in the complementary filter (1-this) is the measured joint angles
    'move': True, #true: walks according to cpg.direction, false: stands in place (will continue to stabilize); leave to true for CPG convergence
    'xmk': xmk, #Snake Monster Kinematics object
    'pose': np.eye(3), #%SO(3) describing ground frame w.r.t world frame
    'G': np.eye(3), #SO(3) describing ground frame w.r.t world frame
    'tp': np.zeros([4,1]),
    'dynY': 0,
    'vY': 0,
    'direction' : 'forward',
    'fullStepLength' : 20000,
    't' : 0
}

cx = np.array([-1/6, -1/6, 0, 0, 0, 0]) * pi
cy = np.array([0, 0, 0, 0, 0, 0]) * pi

cpg['eePos'] = np.vstack((cpg['nomX'],cpg['nomY'], -cpg['h'] * np.ones([1,6]) )) # R: Compute the EE positions in body frame
ang = cpg['xmk'].getLegIK(cpg['eePos']) #R: This gives the angles corresponding to each of the joints
cpg['nomOffset'] = np.reshape(ang[0:18], [6, 3]).T
cpg['nomOffset'] = cpg['nomOffset'] * cpg['scaling']

# the distance between foot ground trajectory (TUNEABLE)
cpg['foothold_offsetY'] = np.array([0.33145, - 0.33145,   0.5125, - 0.5125,   0.33105, - 0.33105])

# the distance between foot ground trajectory to base actuator
dist = cpg['foothold_offsetY'] - cpg['s1OffsetY']

# calculate the respective to cx to make stride length of 6 legs to be equal
a = calculate_a(cpg, cx, dist)

cpg['a'] = a * cpg['scaling']
cpg['b'] = cpg['b'] * cpg['scaling']
cpg['cx'] = cx * cpg['scaling']
cpg['cy'] = cy * cpg['scaling']

cpg['K'] = np.array( [[0, -1, -1,  1,  1, -1],
                     [-1,  0,  1, -1, -1,  1],
                     [-1,  1,  0, -1, -1,  1],
                     [ 1, -1, -1,  0,  1, -1],
                     [ 1, -1, -1,  1,  0, -1],
                     [-1,  1,  1, -1, -1,  0]])

# Initialize the x and y values of the cpg cycle
cpg['x'][0, :] = (a * np.array([1, -1, -1, 1, 1, -1]) + cx) * cpg['scaling']
cpg['y'][0, :] = np.zeros(6)
dt = 0.01  # CPG frequency

frame = 0

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    cpg = updateCPGStance(cpg, cpg['t'])
    cpg, positions = CPG(cpg, cpg['t'], dt)

    cpg['t'] += 1

    if cpg['t'] > (cpg['initLength'] + 1):
        cpg['move'] = True
        cpg_position = cpg['legs'][0, :]
        print(cpg_position)
        print("数据类型", type(cpg_position))  # 打印数组数据类型
        print("数组元素数据类型：", cpg_position.dtype)  # 打印数组元素数据类型
        print("数组元素总数：", cpg_position.size)  # 打印数组尺寸，即数组元素总数
        print("数组形状：", cpg_position.shape)  # 打印数组形状
        print("数组的维度数目", cpg_position.ndim)  # 打印数组的维度数目
        # leg.num correction due to urdf file
        dof_states=[]
        for i in cpg_position:
            tup = (i, 0)
            dof_states.append(tup)

    if args.show_axis:
        gym.clear_lines(viewer)

    # clone actor state in all of the environments
    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
