import torch
from sam2.build_sam import build_sam2_camera_predictor
import os
import cv2

current_file_path = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(current_file_path))

class MaskTracker:
    def __init__(self,):
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        checkpoint_path = f"{project_dir}/third_party/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
        self.camera_predictor = build_sam2_camera_predictor(model_cfg,checkpoint_path,device='cuda')
        self.initialized = False

    def init_sam(self,init_image):
        self.camera_predictor.load_first_frame(init_image)
        points, labels = self.select_point_for_image(init_image
        )
        _,_,_ = self.camera_predictor.add_new_prompt(
            frame_idx=0,
            obj_id=(0),
            points=points,
            labels=labels
        )
        self.initialized = True
        return
    
    def track(self, image):
        if not self.initialized:
            self.init_sam(image)
        obj_id, out_mask_logits = self.camera_predictor.track(image)
        mask = out_mask_logits[0][0] > 0.0

        return mask

    def select_point_for_image(self,image):
        def select_point_cv2(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                positive.append((x, y))
                cv2.circle(img_visual, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow("Image", img_visual)
            if event == cv2.EVENT_RBUTTONDOWN:
                negative.append((x, y))
                cv2.circle(img_visual, (x, y), 3, (255, 0, 0), -1)
                cv2.imshow("Image", img_visual)
        positive = []
        negative = []
        img_visual = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        print('please select points, press esc to finish')
        cv2.imshow("Image", img_visual)
        cv2.setMouseCallback("Image", select_point_cv2)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # 必须按Esc退出，否则会卡住
                break
        cv2.destroyAllWindows()
        # import ipdb;ipdb.set_trace()
        points = positive + negative
        labels = [1]*len(positive) + [0]*len(negative)
        # exit()
        return points, labels