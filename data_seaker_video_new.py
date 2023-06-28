### 
### This file is used to load DANNCE dataset. 
### Copyright @ Liang An 2022
### 
import pickle 
import numpy as np 
import cv2 
import json 
import os 
from utils import * 


class DataSeakerDet():
    def __init__(self): 
        self.data_dir = "data/markerless_mouse_1_nerf/"
        self.raw_caps = []
        self.bg_caps  = []  
        self.views_to_use = [0,1,2,3,4,5]
        
        for k in range(6): 
            video_path = os.path.join(self.data_dir, "videos_undist", "{}.mp4".format(k))
            cap = cv2.VideoCapture(video_path) 
            self.raw_caps.append(cap) 
            mask_path = os.path.join(self.data_dir, "simpleclick_undist", "{}.mp4".format(k)) 
            cap = cv2.VideoCapture(mask_path) 
            self.bg_caps.append(cap) 
        self.totalframes = 18000 

        with open(os.path.join(self.data_dir, "new_cam.pkl"), 'rb') as f: 
            self.cams_dict = pickle.load(f)
        for camid in range(6): 
            self.cams_dict[camid]['T'] = np.expand_dims(self.cams_dict[camid]['T'], 0)
            self.cams_dict[camid]['R'] = self.cams_dict[camid]['R'].T
            self.cams_dict[camid]['K'] = self.cams_dict[camid]['K'].T
        self.cams_dict_out = []
        for camid in self.views_to_use: 
            self.cams_dict_out.append(self.cams_dict[camid])

        self.keypoint_num = 22 

        self.poses2d =[] 
        for camid in range(6): 
            filename = os.path.join(self.data_dir, "keypoints2d_undist", "result_view_{}.pkl".format(camid) )
            with open(filename, "rb") as f: 
                data = pickle.load(f) # (18000, 22 ,3)
            self.poses2d.append(data) 
        self.last_index = -1

    def fetch(self, index, with_img = True):
        if index >= self.totalframes: 
            print("error: index out of range.") 
            return None
        ## 2. read rgb images 
        imgs = []
        if with_img:
            if index - self.last_index == 1: 
                for camid in self.views_to_use:
                    _,img = self.raw_caps[camid].read() 
                    imgs.append(img) 
            else: 
                for camid in self.views_to_use: 
                    self.raw_caps[camid].set(cv2.CAP_PROP_POS_FRAMES, index) 
                    _, img = self.raw_caps[camid].read() 
                    imgs.append(img) 

        ## 3. read masks
        bgs = []
        if index - self.last_index == 1: 
            for camid in self.views_to_use:
                _, bg = self.bg_caps[camid].read() 
                bgs.append(bg.astype(np.float32)[:,:,0] / 255) 
            self.last_index = index
        else: 
            for camid in self.views_to_use: 
                self.bg_caps[camid].set(cv2.CAP_PROP_POS_FRAMES, index) 
                _, bg = self.bg_caps[camid].read() 
       
                bgs.append(bg.astype(np.float32)[:,:,0] / 255) 
            self.last_index = index     
    
        ## 4: fetch keypoints 
        all_keypoints =[]
        for camid in self.views_to_use: 
            data = self.poses2d[camid][index]
            w = data[:,2]
            data[w<0.25,:] = 0
            all_keypoints.append(data) 

        output = {
            "imgs": imgs, 
            "bgs" : bgs, 
            "label2d": np.asarray(all_keypoints) 
        }

        return output

    def visualize_data(self, index):
        output = self.fetch(index)
        cams = self.cams_dict_out
        camN = len(cams) 
        imgs = output["imgs"]
        label2ds = output["label2d"]
        renders = [] 
        print(camN)
        for camid in range(camN):
            img2 = draw_keypoints(imgs[camid], label2ds[camid], bones, is_draw_bone=True)
            renders.append(img2) 
        
        outputimg = pack_images(renders) 
        cv2.namedWindow("2d", cv2.WINDOW_NORMAL) 
        cv2.imshow("2d", outputimg) 
        cv2.waitKey() 
        cv2.destroyAllWindows() 

if __name__ == "__main__": 
    loader = DataSeakerDet() 
    # output = loader.fetch(8) 

    loader.visualize_data(14500) 