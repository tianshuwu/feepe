import trimesh
from utils import transform_pointcloud, transform_pointcloud_multiT
import copy
import numpy as np

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


class MemorypoolOptimizer:
    def __init__(self,mesh):
        self.mesh = mesh
        self.memorypool = []
        self.small_mesh_point = trimesh.sample.sample_surface(self.mesh, 128)[0]

        self.support_frame  = None
        self.last_g2c = None
        self.last_g2b = None

        self.renew_size = [30,15]   # 达到30帧时，随机保留15帧

    def optimize(self, match_inlier_k, obj2cam_k, tcp):

        self.memorypool.insert(0,{
            'match_inlier_k': match_inlier_k, # (k,num_inliers,6(template_objspace,scene_camspace))
            'obj2cam_k': np.array(obj2cam_k),        # (k,4,4)
            'tcp': tcp,                        # (4,4)
            'predicted_camspace_point': transform_pointcloud_multiT(self.small_mesh_point, np.array(obj2cam_k)), # (k,128,3)
        })

        # step1: vote to remove mispredictions due to symmetry
        select_ref_view = self.vote()
        if self.last_g2c is None:
            self.last_g2c = self.memorypool[0]['obj2cam_k'][0]
            self.last_g2b = self.memorypool[0]['tcp']
        current_g2c_init = self.last_g2c @ np.linalg.inv(self.last_g2b) @ self.memorypool[0]['tcp']
        
        # step2: 全部转换到同一帧下，辅助当前帧进行优化
        q = R.from_matrix(current_g2c_init[:3,:3]).as_quat()
        t = current_g2c_init[:3,3]
        x0 = np.concatenate([q, t])
        p_obj_list = [np.concatenate([self.memorypool[i]['match_inlier_k'][j][:,:3] for j in select_ref_view[i]]) for i in range(len(self.memorypool))]
        p_cam_list = [np.concatenate([self.memorypool[i]['match_inlier_k'][j][:,3:] for j in select_ref_view[i]]) for i in range(len(self.memorypool))]
        g2b_list = [self.memorypool[i]['tcp'] for i in range(len(self.memorypool))]

        result = least_squares(residual_fn, x0, args=(p_obj_list, p_cam_list, g2b_list), method='lm', verbose=0)
        q_opt = result.x[0:4] / np.linalg.norm(result.x[0:4])
        t_opt = result.x[4:7]
        T_opt = np.eye(4)
        T_opt[:3,:3] = R.from_quat(q_opt).as_matrix()
        T_opt[:3,3]  = t_opt

        self.last_g2c = T_opt
        self.last_g2b = self.memorypool[0]['tcp']

        if len(self.memorypool) > self.renew_size[0]:
            indices = np.random.choice(len(self.memorypool), self.renew_size[1], replace=False)
            self.memorypool = [self.memorypool[i] for i in indices]

        return T_opt

    def vote(self):
        if self.support_frame is None:
            self.support_frame = self.memorypool[0]
        else:
            if len(self.memorypool[0]['match_inlier_k'][0]) > len(self.support_frame['match_inlier_k'][0]):
                self.support_frame = copy.deepcopy(self.memorypool[0])

        # use history frame to vote to choose a best candidate pose in k pose for the support frame
        best_ref_indices_in_support = []
        for i in range(len(self.memorypool)):

            transfertosupportspace = np.array([g2c @ np.linalg.inv(self.memorypool[i]['tcp']) @ self.support_frame['tcp'] @ np.linalg.inv(g2c) for g2c in self.memorypool[i]['obj2cam_k']]) # (k,4,4)

            point_tranfertosupportspace = np.array([transform_pointcloud_multiT(point,transfertosupportspace) for point in self.memorypool[i]['predicted_camspace_point']]) # (k,128,3)

            dis_matrix = np.linalg.norm(point_tranfertosupportspace[:,None,:,:] - self.support_frame['predicted_camspace_point'][None,:,:,:], axis=-1).mean(axis=-1) # (k,k)
            best_ref_indices_in_support.append(np.argmin(dis_matrix, axis=1))

        # 找到元素个数最多的标签
        best_ref_indices_in_support = np.array(best_ref_indices_in_support)
        best_ref_indices_in_support = best_ref_indices_in_support.flatten()
        unique, counts = np.unique(best_ref_indices_in_support, return_counts=True)
        largest_class_label = unique[np.argmax(counts)]

        # 反推到每一帧的旋转上
        cam2base_support = self.support_frame['tcp'] @ np.linalg.inv(self.support_frame['obj2cam_k'][largest_class_label])
        select_ref_view = []
        for i in range(len(self.memorypool)):
            obj2cam_ref = np.linalg.inv(cam2base_support) @ self.memorypool[i]['tcp']

            # 计算旋转之间的差值
            R_rel = self.memorypool[i]['obj2cam_k'] @ obj2cam_ref.T
            # 计算每个R_rel的旋转角度
            angles_rad = np.arccos(np.clip((np.trace(R_rel, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0))
            angles_deg = np.degrees(angles_rad)
            # 获取小于45度的下标
            indices = np.where(angles_deg < 45)[0]
            if len(indices) == 0:
                min_index = np.argmin(angles_deg)
                indices = np.array([min_index])
            select_ref_view.append(indices)
        return select_ref_view
    
def residual_fn(x, p_obj_list, p_cam_list,g2b_list):
    # x = [x, y, z, w, x,y,z]
    # p_obj_list: [tx(N,3)]
    # p_cam_list: [tx(N,3)]
    # g2b_list: [tx(4,4)]

    q = x[0:4]
    t = x[4:7]
    q = q / np.linalg.norm(q)
    g2c_new = np.eye(4)
    g2c_new[:3,:3] = R.from_quat(q).as_matrix()
    g2c_new[:3,3] = t
    res = []
    for i in range(len(g2b_list)):
        g2c_i = g2c_new @ np.linalg.inv(g2b_list[0]) @ g2b_list[i]
        res.append(p_cam_list[i] - transform_pointcloud(p_obj_list[i], g2c_i))
    return np.concatenate(res).ravel()