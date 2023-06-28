import numpy as np 

mouse_22_bones = [
    [0,2], [1,2], 
    [2,3],[3,4],[4,5],[5,6],[6,7],
    [8,9], [9,10], [10,11], [11,3],
    [12,13], [13,14], [14,15], [15,3],
    [16,17],[17,18],[18,5],
    [19,20],[20,21],[21,5]
] 
mouse_22_bone_color_index = [ 
    0,0,
    3,3,3,3,3,
    1,1,1,1,
    2,2,2,2,
    4,4,4,
    5,5,5
]
# RGB
mouse_22_colormap = np.loadtxt("colormaps/anliang_paper.txt", dtype=np.uint8)

mouse_22_joint_color_index = [ 
    0,0,0,
    3,3,3,3,3,
    1,1,1,1,
    2,2,2,2,
    4,4,4,
    5,5,5
]

keypoint_names = [
    "left_ear_tip",# 0
    "right_ear_tip",# 1
    "nose", # 2
    "neck", # 3
    "body_middle", # 4
    "tail_root", # 5
    "tail_middle", # 6
    "tail_end", # 7
    "left_paw", # 8
    "left_paw_end", # 9
    "left_elbow", # 10
    "left_shoulder", # 11
    "right_paw", # 12 
    "right_paw_end", # 13
    "right_elbow", # 14
    "right_shoulder", # 15
    "left_foot", # 16
    "left_knee", # 17
    "left_hip", # 18 
    "right_foot", # 19 
    "right_knee", # 20
    "right_hip" # 21 
],