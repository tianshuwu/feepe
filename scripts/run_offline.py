from matcher import Matcher
from mask_tracker import MaskTracker
from optimizer import MemorypoolOptimizer
import numpy as np
import os
import time
import cv2
import json
current_file_path = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(current_file_path))


class DataLoaderOffline:
    def __init__(self,data_dir):
        self.data = np.load(data_dir, allow_pickle=True)

        self.depth_list = self.data['depth']
        self.rgb_list = self.data['rgb']
        self.tcp_list = self.data['poses']
        self.intrinsics = self.data['intrinsics'].item()

        self.mask_tracker = MaskTracker()


    def get_frame(self, index):
        depth = self.depth_list[index][..., None]      # h,w,1, m,np
        rgb = self.rgb_list[index]          # h,w,3, np.uint8
        tcp = self.tcp_list[index]          # 4,4 np
        mask = self.mask_tracker.track(rgb).unsqueeze(-1) # h,w,1 bool,tensor

        return {'depth':depth, 'rgb':rgb, 'tcp':tcp,'mask':mask, 'intrinsics':self.intrinsics}


if __name__ == "__main__":
    visualization = True
    data_dir = './data/franka_record_data.npz'  
    data_loader = DataLoaderOffline(data_dir)

    matcher = Matcher(gripper_name='panda', cad_name='model.obj')
    output_video_writer = None
    
    optimizer = MemorypoolOptimizer(matcher.mesh)
    for i in range(len(data_loader.rgb_list)):
        frame = data_loader.get_frame(i)

        time_start = time.time()
        match_inlier_k,pose_k = matcher.match(frame)
        time_match = time.time()
        T_opt = optimizer.optimize(match_inlier_k, pose_k, frame['tcp'])    # this is optimized gripper2cam pose result
        time_opt = time.time()
        print(f"Matching time: {time_match - time_start:.3f}s, Optimization time: {time_opt - time_match:.4f}s")



        if visualization:
            # Visualize the mask
            mask_color = cv2.applyColorMap(frame['mask'].cpu().numpy().astype(np.uint8)*255, cv2.COLORMAP_JET)
            mask_vis = cv2.addWeighted(cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR), 0.7, mask_color, 0.3, 0)
            mask_vis = cv2.putText(mask_vis, 'Mask', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

            # Visualize the depth
            depth_vis = (np.clip(np.squeeze(frame['depth'],axis=2),0,3) / 3.0 * 255.0).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_vis = cv2.putText(depth_vis, 'Depth', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

            # Visualize the no optimized result
            no_optimized_vis = matcher.render_pyrender(frame,pose_k[0])
            no_optimized_vis = cv2.addWeighted(cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR),1.0, no_optimized_vis, 1.0, 0)
            no_optimized_vis = cv2.putText(no_optimized_vis, 'Before Optimization', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

            # Visualize the optimized result
            optimized_vis = matcher.render_pyrender(frame,T_opt)
            optimized_vis = cv2.addWeighted(cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR), 1.0, optimized_vis, 1.0, 0)
            optimized_vis = cv2.putText(optimized_vis, 'After Optimization', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

            top_row = np.concatenate((mask_vis, depth_vis), axis=1)
            bottom_row = np.concatenate((no_optimized_vis, optimized_vis), axis=1)
            combined_vis = np.concatenate((top_row, bottom_row), axis=0)

            if output_video_writer is None:
                frame_size = (optimized_vis.shape[1], optimized_vis.shape[0])
                output_video_writer = cv2.VideoWriter('./data/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 12, frame_size)
            output_video_writer.write(optimized_vis)

            cv2.imshow("Combined Visualization", combined_vis)
            cv2.waitKey(1)
    
    if output_video_writer is not None:
        output_video_writer.release()
    cv2.destroyAllWindows()

    cam2base = frame['tcp'] @ np.linalg.inv(T_opt)  # cam2base = gripper2base @ cam2gripper
    print('cam2base:\n',cam2base)