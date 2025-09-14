#!/usr/bin/env python3
import threading
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from frankapy import FrankaArm
from queue import Queue
from autolab_core import RigidTransform
import os

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy = color_intrinsics.fx, color_intrinsics.fy
cx, cy = color_intrinsics.ppx, color_intrinsics.ppy
intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

align_to = rs.stream.color
align = rs.align(align_to)

frame_queue = Queue(maxsize=1)

def camera_thread():
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0

        if frame_queue.full():
            frame_queue.get()
        frame_queue.put((color, depth, time.time()))

cam_thread = threading.Thread(target=camera_thread, daemon=True)
cam_thread.start()

fa = FrankaArm()
fa.reset_joints()
time.sleep(2)


T_init = fa.get_pose().matrix # numpy (4,4)

steps = 64
motion_radius = 0.1
rot_angle = 60 * np.pi / 180

recorded_poses = []
recorded_rgb = []
recorded_depth = []
recorded_timestamps = []

for i in range(steps):

    dx = motion_radius * np.sin(2 * np.pi * i / steps)
    dy = motion_radius * np.cos(2 * np.pi * i / steps)

    dT = np.eye(4)
    dT[0,3] = dx
    dT[1,3] = dy

    theta = rot_angle * np.sin(2 * np.pi * i / steps)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]])
    dT[:3,:3] = Rz

    T_target = T_init @ dT
    T_target = RigidTransform(
    rotation=T_target[:3,:3],
    translation=T_target[:3,3],
    from_frame='franka_tool', to_frame='world'
)

    fa.goto_pose(T_target, duration=0.2, use_impedance=True)

    if not frame_queue.empty():
        color, depth, ts = frame_queue.get()
        recorded_rgb.append(color.copy())
        recorded_depth.append(depth.copy())
        recorded_timestamps.append(ts)

    recorded_poses.append(fa.get_pose().matrix)
    time.sleep(0.05)

np.savez("data/franka_record_data.npz",
    poses=np.array(recorded_poses),
    rgb=np.array(recorded_rgb),
    depth=np.array(recorded_depth),
    timestamps=np.array(recorded_timestamps),
    intrinsics=intrinsics)

print("data saved to data/franka_record_data.npz")

pipeline.stop()