import open3d as o3d
import numpy as np
import torch 
import torchvision.transforms as transforms
import torch.nn.functional as F

def sample_points_from_mesh(path, n_pts):
    mesh = o3d.io.read_triangle_mesh(path)
    pcd = mesh.sample_points_uniformly(number_of_points=n_pts)
    points = np.asarray(pcd.points)

    return points

def transform_pointcloud(pointcloud, transformation_matrix):
    homogeneous_points = np.hstack((pointcloud, np.ones((pointcloud.shape[0], 1))))
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T
    transformed_points = transformed_points[:, :3] / transformed_points[:, 3][:, np.newaxis]
    
    return transformed_points


def transform_pointcloud_multiT(pointcloud, transformation_matrixs):
    N, D = pointcloud.shape
    N_trans, D1, D2 = transformation_matrixs.shape
    assert D == 3
    assert D1 == 4 and D2 == 4

    # [N, 4]
    ones = np.ones((N, 1), dtype=pointcloud.dtype)
    homogeneous_points = np.concatenate([pointcloud, ones], axis=1)

    # [N_trans, N, 4]
    homogeneous_points = np.broadcast_to(homogeneous_points, (N_trans, N, 4))

    # [N_trans, 4, 4] × [N_trans, N, 4, 1] -> [N_trans, N, 4]
    transformed_points = transformation_matrixs @ homogeneous_points.transpose(0, 2, 1)
    transformed_points = transformed_points.transpose(0, 2, 1)  # [N_trans, N, 4]

    # Convert back to Euclidean coordinates
    transformed_points = transformed_points[:, :, :3] / transformed_points[:, :, 3:4]

    return transformed_points


def depth_map_to_pointcloud(depth_map, mask, intrinsics):
    # depth_map : h,w
    # mask : h,w

    # Get dimensions
    H, W = depth_map.shape

    if mask is not None:
        depth_map[mask == 0] = -1

    # Unpack intrinsic matrix
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    # Create grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # Convert pixel coordinates to camera coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map

    # Reshape to (B*S, H*W)
    x = np.reshape(x, (-1))
    y = np.reshape(y, (-1))
    z = np.reshape(z, (-1))

    # Stack into point cloud
    pointcloud = np.stack((x, y, z), axis=-1)

    pointcloud = pointcloud[pointcloud[:, 2] > 0]
    return pointcloud


def prepare_images(rgbs, depths, masks, size):
    if len(rgbs.shape) == 3:
        rgbs = torch.tensor(rgbs).unsqueeze(0)
        depths = torch.tensor(depths).unsqueeze(0)
        masks = torch.tensor(masks).unsqueeze(0)
    rgbs = rgbs.permute(0, 3, 1, 2)
    depths = depths.permute(0, 3, 1, 2)
    masks = masks.permute(0, 3, 1, 2)
    # print(rgbs.shape,depths.shape,masks.shape)

    rgbs = rgbs / 255.0  # B, 3, H, W
    # depths = images[:,3:4] # B, 1, H, W
    # masks = images[:,4:5] # B, 1, H, W
    B = rgbs.shape[0]
    bboxes = get_2dbboxes(masks[:, 0])  # B, 4
    cropped_rgbs = torch.zeros((B, 3, size, size))
    cropped_masks = torch.zeros((B, 1, size, size))
    for b in range(B):
        y_min, x_min, y_max, x_max = bboxes[b, 0], bboxes[b, 1], bboxes[b, 2], bboxes[b, 3]  # bbox是y,x的形式？

        cropped_rgb = rgbs[b:b + 1, :, y_min:y_max, x_min:x_max]
        cropped_mask = masks[b:b + 1, :, y_min:y_max, x_min:x_max]
        cropped_rgb = F.interpolate(cropped_rgb, size=(size, size), mode="bilinear")
        cropped_mask = F.interpolate(cropped_mask.to(torch.uint8), size=(size, size), mode="nearest")
        cropped_rgbs[b:b + 1] = cropped_rgb
        cropped_masks[b:b + 1] = cropped_mask

    # no_transform = cropped_rgbs  # Image.fromarray(cropped_rgbs[0].cpu().numpy().astype(int)*255)
    cropped_rgbs = transforms.Compose([
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # imagenet defaults
    ])(cropped_rgbs)
    # cropped_rgbs = cropped_rgbs

    cropped_rgbs[cropped_masks.repeat(1, 3, 1, 1) == 0] = 0  # Mask out background

    bboxes = torch.tensor(bboxes)

    return cropped_rgbs, cropped_masks, bboxes#, no_transform  # b,?,h,w


def get_2dbboxes(masks):

    assert len(masks.shape) == 3     # make bbox square
    B, H, W = masks.shape
    # Initialize tensor to store bounding boxes
    bboxes = torch.zeros((B, 4))
    # Iterate over batch dimension
    for b in range(B):
        # Find coordinates of non-zero elements in the mask
        non_zero_coords = torch.nonzero(masks[b].float())
        #print(non_zero_coords.shape)
        #print(non_zero_coords)

        # Extract bounding box coordinates
        ymin = non_zero_coords[:, 0].min()
        ymax = non_zero_coords[:, 0].max() + 1
        xmin = non_zero_coords[:, 1].min()
        xmax = non_zero_coords[:, 1].max() + 1

        # add function: expand bbox

        w = xmax-xmin
        h = ymax-ymin
        if h > w:
            dif = h-w
            xmin = xmin-dif//2
            xmax = xmax+dif//2
            if xmin < 0:
                xmax = xmax - xmin
                xmin = 0
            if xmax > W:
                xmin = xmin-xmax+W
                xmax = W
        elif w>h :
            dif = w-h
            ymin = ymin-dif/2
            ymax = ymax+dif/2
            if ymin < 0:
                ymax = ymax - ymin
                ymin = 0
            if ymax > H:
                ymin = ymin-ymax+H
                ymax = H
        ymin = max(ymin,0)
        xmin = max(xmin,0)
        ymax = min(ymax, H)
        xmax = min(xmax, W)
        # Store bounding box coordinates
        bboxes[b] = torch.tensor([ymin, xmin, ymax, xmax])
        bboxes = torch.clamp(bboxes,0)
    
    return bboxes.int().numpy()



class PCALowrank():
    def __init__(self):
        self.V = None
        self.mean = None
        # self.std = None

    def fit(self,X,q=256):
        X_mean = torch.mean(X,dim=0)
        # X_std = torch.std(X,dim=0)
        X_centered = (X - X_mean)# / X_std
        U,S,V = torch.pca_lowrank(X_centered,q=q)
        self.V = V
        self.mean = X_mean
        # self.std = X_std

    def transform(self,X):
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        else:
            assert len(X.shape) == 2
        X_centered = (X - self.mean)# / self.std
        return torch.matmul(X_centered,self.V)