import torch
import os 
import sys
import glob
import open3d as o3d
import torch.nn.functional as F
import numpy as np
import cv2
import json
from utils import prepare_images,transform_pointcloud
from concurrent.futures import ThreadPoolExecutor

from sklearn.cluster import KMeans,MiniBatchKMeans
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from tqdm import tqdm
import pyrender
import trimesh
import time

current_file_path = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(current_file_path))


class Matcher:
    def __init__(self, gripper_name, cad_name):
        template_dir = f'{project_dir}/template/{gripper_name}'

        self.cad_path = f'{template_dir}/model/{cad_name}'
        self.mesh = trimesh.load(self.cad_path)
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate([geom for geom in self.mesh.geometry.values()])
        self.pyrender_mesh = pyrender.Mesh.from_trimesh(self.mesh)

        self.max_point_size = 512
        self.k = 5
        self.init_dino()
        self.init_template(template_dir)
        self.init_bow(template_dir)


    def init_dino(self):
        self.dino = torch.hub.load(repo_or_dir=f"{project_dir}/third_party/dinov2",source="local", model="dinov2_vits14_reg", pretrained=False)
        self.dino.load_state_dict(torch.load(f"{project_dir}/third_party/dinov2/dinov2_vits14_reg4_pretrain.pth"))
        self.dino.eval()
        self.dino.to('cuda')
        self.dino_process_size = 224
    
    def init_template(self,template_dir):
        # for feature-based matching
        self.template_feature = [] # num_ref,c
        self.template_point_camspace = [] # num_ref,3
        self.obj2cam = [] # num_ref,4x4
        self.template_ids = [path.split('/')[-1][:6] for path in sorted(glob.glob(f'{template_dir}/render' + "/*rgb.png"))] # num_ref
        for template_id in self.template_ids:
            if not os.path.exists(f'{template_dir}/feature/{template_id}_feature.npy'):
                os.makedirs(f'{template_dir}/feature',exist_ok=True)
                print('getting feature for template',template_id)
                with open(f'{template_dir}/render/camera_intrinsics.json', 'r') as f:
                    intrinsics = json.load(f)
                frame = {
                    'rgb':cv2.imread(f'{template_dir}/render/{template_id}_rgb.png'),
                    'depth':cv2.imread(f'{template_dir}/render/{template_id}_depth.png',cv2.IMREAD_ANYDEPTH)[:,:,None] / 1000.0, 
                    'mask':cv2.imread(f'{template_dir}/render/{template_id}_mask.png',cv2.IMREAD_ANYDEPTH)[:,:,None],
                    'intrinsics':intrinsics
                }
                point,feat_point = self.get_feature(frame)       # n,3; n,c
                np.save(f'{template_dir}/feature/{template_id}_point.npy',point.cpu().numpy())
                np.save(f'{template_dir}/feature/{template_id}_feature.npy',feat_point.cpu().numpy())

            self.template_feature.append(np.load(f'{template_dir}/feature/{template_id}_feature.npy'))
            self.template_point_camspace.append(np.load(f'{template_dir}/feature/{template_id}_point.npy'))
            self.obj2cam.append(np.load(f'{template_dir}/render/{template_id}_o2c.npy'))

    def init_bow(self,template_dir):
        # for topk refs selection
        
        self.num_words = 2048
        if not os.path.exists(f'{template_dir}/feature/vision_words.npy'):
            print('generating vision words and template descriptors')#, might take a few minutes')
            kmeans = MiniBatchKMeans(n_clusters=self.num_words,batch_size=1024)
            # kmeans = KMeans(n_clusters=self.num_words)
            kmeans.fit(np.concatenate(self.template_feature,axis=0)) 
            centers = kmeans.cluster_centers_  # num_words,c

            np.save(f'{template_dir}/feature/vision_words.npy',centers)
            self.vision_words = torch.from_numpy(np.load(f'{template_dir}/feature/vision_words.npy'))

            ref_bags = []
            for template_feature in self.template_feature:
                ref_bags.append(self.get_descriptor(torch.from_numpy(template_feature))[0])
            ref_bags = torch.stack(ref_bags)  # num_ref,num_words
            
            np.save(f'{template_dir}/feature/template_descriptors.npy',ref_bags.cpu().numpy())

        self.vision_words = torch.from_numpy(np.load(f'{template_dir}/feature/vision_words.npy'))        # num_words,c
        self.template_descriptors = torch.from_numpy(np.load(f'{template_dir}/feature/template_descriptors.npy'))     # num_ref,num_words


    def match(self,frame):
        # 1. get feature
        point,feat_point = self.get_feature(frame)       # n,3; n,c

        point, feat_point = point.cpu(), feat_point.cpu()  

        # 2. select topk refs
        descriptor = self.get_descriptor(feat_point)    # 1,2048
        similarity = pairwise_cosine_similarity(descriptor,self.template_descriptors) # 1,num_template
        _,sorted_indices = torch.sort(similarity,dim=1,descending=True)
        selected_indices = sorted_indices[0,:self.k]  # k

        # 3. get k matches

        # multithread
        with ThreadPoolExecutor(max_workers=self.k) as executor:
            futures = [executor.submit(self.match_one, tid, point, feat_point) for tid in selected_indices]
            results = [f.result() for f in futures]

        match_inlier_list, obj2cam_list = zip(*results) # k,point_inlier,6(template,scene); k,4x4
        return list(match_inlier_list), list(obj2cam_list)
    
        # # single thread
        # match_inlier_list = []
        # obj2cam_list = []
        # for template_id in selected_indices:

        #     point_template_camspace = self.template_point_camspace[template_id]       # what coordinate? obj space
        #     feat_point_template = self.template_feature[template_id]
        #     obj2cam_template = self.obj2cam[template_id]
        #     point_template_objspace = transform_pointcloud(point_template_camspace,np.linalg.inv(obj2cam_template))   # n,3

        #     pcd_source = o3d.geometry.PointCloud()
        #     pcd_source.points = o3d.utility.Vector3dVector(point_template_objspace)
        #     feature_source = o3d.pipelines.registration.Feature()
        #     feature_source.data = feat_point_template.T#.cpu().numpy()

        #     pcd_target = o3d.geometry.PointCloud()
        #     pcd_target.points = o3d.utility.Vector3dVector(point.numpy())
        #     feature_target = o3d.pipelines.registration.Feature()
        #     feature_target.data = feat_point.numpy().T

        #     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        #         pcd_source,
        #         pcd_target,
        #         feature_source,
        #         feature_target,
        #         False,
        #         0.01,
        #         checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        #                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.01)],
        #         # criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000,1.0)
        #     )

        #     correspondence_set = np.asarray(result.correspondence_set)

        #     match_inlier_list.append(np.concatenate([point_template_objspace[correspondence_set[:,0]], point[correspondence_set[:,1]]],axis=1))  # point_inlier,6(template,scene)
        #     obj2cam_list.append(result.transformation)  # 4x4
        # return match_inlier_list,obj2cam_list



        
    def match_one(self,template_id, point, feat_point,):
        point_template_camspace = self.template_point_camspace[template_id]       # what coordinate? obj space
        feat_point_template = self.template_feature[template_id]
        obj2cam_template = self.obj2cam[template_id]
        point_template_objspace = transform_pointcloud(point_template_camspace,np.linalg.inv(obj2cam_template))   # n,3

        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(point_template_objspace)
        feature_source = o3d.pipelines.registration.Feature()
        feature_source.data = feat_point_template.T#.cpu().numpy()

        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(point.numpy())
        feature_target = o3d.pipelines.registration.Feature()
        feature_target.data = feat_point.numpy().T

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_source,
            pcd_target,
            feature_source,
            feature_target,
            False,
            0.01,
            checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.01)],
            # criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(10000,1.0)
        )

        correspondence_set = np.asarray(result.correspondence_set)

        match_inlier = np.concatenate([point_template_objspace[correspondence_set[:,0]], point[correspondence_set[:,1]]],axis=1)  # point_inlier,6(template,scene)
        obj2cam = result.transformation  # 4x4
        return match_inlier,obj2cam

        

    def get_feature(self, frame):
        rgb,depth,mask,intrinsics = frame['rgb'],frame['depth'],frame['mask'],frame['intrinsics']
        rgb,depth = torch.from_numpy(rgb).float().to('cuda'),torch.from_numpy(depth).float().to('cuda') 

        # todo: preprocess rgb
        cropped_rgb, cropped_mask, bbox = prepare_images(rgb,depth,mask,self.dino_process_size)    # 1,...

        with torch.no_grad():
            # forward feature
            feat_dino = self.dino.get_intermediate_layers(cropped_rgb.to('cuda'),n=[len(self.dino.blocks)-1])[0]    # 1,l,c
            feat_dino = feat_dino.permute(0,2,1).reshape(1,-1,self.dino_process_size//14,self.dino_process_size//14) # 1,c,patch_h,patch_w

            # get box feature map
            ymin, xmin, ymax, xmax = bbox[0]
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            feat_box = F.interpolate(feat_dino,(bbox_height,bbox_width),mode='bilinear')   # 1,c,box_h,box_w

            # get image feature
            feat_image = torch.zeros((1,feat_box.shape[1],rgb.shape[0],rgb.shape[1]),device='cuda')
            feat_image[:, :, ymin:ymax, xmin:xmax] = feat_box   # 1,c,h,w

            # get point feature
            depth[mask == 0] = -1
            u, v = torch.meshgrid(torch.arange(rgb.shape[1],device='cuda'), torch.arange(rgb.shape[0],device='cuda'),indexing='xy')
            x = ((u - intrinsics['cx']) * depth[...,0] / intrinsics['fx']).reshape(-1)
            y = ((v - intrinsics['cy']) * depth[...,0] / intrinsics['fy']).reshape(-1)
            z = depth.reshape(-1)   # h*w
            point = torch.stack([x,y,z],dim=1)[z>0]   # n,3
            feat_point = feat_image[0].reshape(feat_box.shape[1],-1).permute(1,0)[z>0]   # n,c
            feat_point = feat_point / feat_point.norm(dim=1,keepdim=True)

            if point.shape[0] > self.max_point_size:
                indices = torch.randperm(point.shape[0])[:self.max_point_size]
                point = point[indices]
                feat_point = feat_point[indices]

        return point,feat_point     # n,3; n,c
    
    def get_descriptor(self,feat_point):
        words_distance = pairwise_euclidean_distance(feat_point,self.vision_words)  # n, num_words
        sorted_dis, sorted_indices = torch.sort(words_distance, dim=1)

        descriptor = torch.zeros(self.num_words, device = 'cpu')

        for j in range(3):
            indices = sorted_indices[:,j]
            dis = torch.exp(-(sorted_dis[:,j]**2)/200)
            descriptor.index_add_(0, indices, dis)
        descriptor = descriptor.unsqueeze(0)

        return descriptor
    
    def render_pyrender(self,frame,obj_pose):

        intrinsic = frame['intrinsics']
        img_size = frame['rgb'].shape[:2]
        cam_pose_init = np.eye(4)
        cam_pose_init[1, 1] = -1
        cam_pose_init[2, 2] = -1

        ambient_light = np.array([1.0, 1.0, 1.0, 1.0])

        scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 1.0]), ambient_light=ambient_light
        )
        light_itensity = 5.0
        light = pyrender.SpotLight(
            color=np.ones(3),
            intensity=light_itensity,
            innerConeAngle=np.pi / 16.0,
            outerConeAngle=np.pi / 6.0,
        )
        scene.add(light, pose=cam_pose_init)

        fx, fy, cx, cy = intrinsic['fx'], intrinsic['fy'], intrinsic['cx'], intrinsic['cy']
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100000
        )

        render_engine = pyrender.OffscreenRenderer(img_size[1], img_size[0])
        cad_node = scene.add(self.pyrender_mesh, pose=np.eye(4), name="cad")


        # 将旋转应用到相机位姿上
        cam_pose = cam_pose_init
        camera_node = scene.add(camera, pose=cam_pose)
        scene.set_pose(cad_node, obj_pose)
        rgb, depth = render_engine.render(scene, pyrender.constants.RenderFlags.RGBA)
        scene.remove_node(camera_node)  # 渲染后移除相机节点
        mask = depth > 0

        # Image.fromarray(np.uint8(rgb)).save(f'')

        return rgb[:,:,:3]