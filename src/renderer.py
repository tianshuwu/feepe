import blenderproc as bproc

import os
import argparse
import json
import cv2
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--gripper_name',type=str)
parser.add_argument('--model_name',type=str)
args = parser.parse_args()

gripper_folder = f'./template/{args.gripper_name}'
obj_fpath = f'{gripper_folder}/model/{args.model_name}'
save_fpath = f'{gripper_folder}/render'
cam_poses_outplanes = np.load(f'./predefined_poses/cam_poses_level0.npy')
if not os.path.exists(save_fpath):
    os.makedirs(save_fpath)
print(f"save to {save_fpath}")

bproc.init()
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.camera.set_resolution(320, 320)
bproc.renderer.set_max_amount_of_samples(1)
bproc.renderer.enable_segmentation_output(map_by=["name", "instance"])


obj = bproc.loader.load_obj(obj_fpath)[0]
obj.set_cp("instance_id", 1)
obj.set_cp("category_id", 1) 
obj.set_name("object")

# blender will change the Y axis and -Z axis of the model, so there will be a obj2world
obj2world = np.eye(4)
obj2world[:3,:3] = [[  1.0000000,  0.0000000,  0.0000000],
[0.0000000, 0, -1],
[0.0000000,  1, 0] ]


bbox_corners = obj.get_bound_box()
bbox_min = bbox_corners.min(axis=0)
bbox_max = bbox_corners.max(axis=0)
diagonal_length = np.linalg.norm(bbox_max - bbox_min)



idx = -1
for cam_poses_outplane in cam_poses_outplanes[:]:

    # set distance on z axis
    cam_poses_outplane[:3,3] = cam_poses_outplane[:3,3] /1000.0 * 3.0*diagonal_length 

    for inplane_rot in range(0, 360, 30):

        # clean up
        bproc.clean_up()
        obj = bproc.loader.load_obj(obj_fpath)[0]
        light1 = bproc.types.Light()


        idx = idx+1
        save_name = str(idx).rjust(6,'0')

        rotation_angle = np.deg2rad(inplane_rot)
        rotation_inplane = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0, 0],
        [np.sin(rotation_angle),  np.cos(rotation_angle), 0, 0],
        [0,                      0,                     1, 0],
        [0,                      0,                     0, 1]
        ])

        cam2world = cam_poses_outplane @ rotation_inplane
        np.save(os.path.join(save_fpath,  f"{save_name}_o2c.npy"), np.linalg.inv(cam2world)@obj2world)

        # for i in [0,2,3]:#range(len(obj.blender_obj.data.materials)):#[0,2,3]:
        #     obj.set_material(i, material)

        # OpenCV camera coordinate -> OpenGL camera coordinate
        cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])
        bproc.camera.add_camera_pose(cam2world)

        # set light
        light1.set_type("POINT")
        light1.set_location([2.5*cam2world[:3, -1][0], 2.5*cam2world[:3, -1][1], 2.5*cam2world[:3, -1][2]])
        light1.set_energy(75)

        data = bproc.renderer.render()

        color_bgr_0 = data["colors"][0]
        color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
        cv2.imwrite(os.path.join(save_fpath,f'{save_name}_rgb'+'.png'), color_bgr_0)


        depth_0 = data["depth"][0]
        depth_uint16 = (depth_0*1000.0).astype(np.uint16) # mm
        depth_image = Image.fromarray(depth_uint16, mode='I;16')
        depth_image.save(os.path.join(save_fpath, f'{save_name}_depth.png'))

        mask = depth_uint16>0
        mask_image = Image.fromarray(np.uint8(mask * 255), mode='L')
        mask_image.save(os.path.join(save_fpath, f'{save_name}_mask.png'))

    camera = bproc.camera.get_intrinsics_as_K_matrix()
    intrinsic = {
        'fx':camera[0][0],
        'fy':camera[1][1],
        'cx':camera[0][2],
        'cy':camera[1][2]
    }
    f = open(os.path.join(save_fpath, "camera_intrinsics.json"), "w")
    f.write(json.dumps(intrinsic))
    f.close()



