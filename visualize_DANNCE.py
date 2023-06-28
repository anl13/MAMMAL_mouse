from struct import pack
import numpy as np 
import pickle
import os 
import cv2 

import sys

from OpenGL.GL import *
import glfw
from glfw.GLFW import *
from numpy.lib.twodim_base import fliplr
import glm 

import matplotlib.pyplot as plt 
from videoio import VideoWriter 
import scipy 
import math
from utils import * 
from tqdm import tqdm 

# Keypoint name: 
# 0: left ear top 
# 1: right ear top 
# 2: nose
# 3: SpineF 
# 4: SpineM 
# 5: tail 
# 6: tail middle 
# 7: tail end
# 8: left hand front
# 9: left hand back
#10: left elbow
#11: left shoulder
#12: right hand front
#13: right hand back
#14: right elbow
#15: right shoulder
#16: left foot 
#17: left knee
#18: left hip 
#19: right foot
#20: right knee
#21: right hip 

DANNCE_folder = "/media/data2/projects/dannce-pytorch" ## set to your own dannce folder 
def visualize_dannce_predict_undist(): 
    ## this version use the pretrained model provided by the author. 2023.04.06
    rawdata = scipy.io.loadmat(os.path.join(DANNCE_folder, "demo/markerless_mouse_1/DANNCE/predict_results/save_data_AVG0.mat"))
    sampleIDs = rawdata["sampleID"].astype(np.int64)[0].tolist() 
    pred = rawdata["pred"] # N, 3, 22 
    data = rawdata["data"] # N, 3, 22
    p_max = rawdata["p_max"] # N, 22 
    # with open("data/markerless_mouse_1/cams.pkl", 'rb') as f: 
        # all_cams = pickle.load(f) 
    with open("data/markerless_mouse_1_nerf/new_cam.pkl", 'rb') as f: 
        all_cams = pickle.load(f) 
    caps = [] 
    for viewid in range(6): 
        videofile = "data/markerless_mouse_1_nerf/videos_undist/{}.mp4".format(viewid)
        cap = cv2.VideoCapture(videofile)
        caps.append(cap) 

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter("mouse_fitting_result/dannce-pytorch.mp4", fourcc, 100, (1728,1024), True) 

    for k in tqdm(range(3000)):  
        frameid = k
        
        data3d = pred[frameid].T / 100
        all_render = [] 
        for camid in range(6):
            cam = all_cams[camid]
            # caps[camid].set(cv2.CAP_PROP_POS_FRAMES, frameid)
            _, img = caps[camid].read() 
            img2 = img.copy() 
            data2d = (data3d@cam["R"].T + cam["T"] / 100)@cam['K'].T
            data2d = data2d[:,0:2] / data2d[:,2:]
            img2 = draw_keypoints(img2, data2d, bones, is_draw_bone=True)
            all_render.append(img2) 
        packed_render=  pack_images(all_render)
        out = cv2.resize(packed_render, (1728, 1024))
        writer.write(out) 
    writer.release() 
    
if __name__ == "__main__":

    visualize_dannce_predict_undist()