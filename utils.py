import cv2 
import numpy as np 
import math 
import torch 

## for dannce labeling (22 keypoints)
colormap = np.loadtxt("colormaps/anliang_paper.txt", dtype=np.uint8)
bones = [
    [0,2], [1,2], 
    [2,3],[3,4],[4,5],[5,6],[6,7],
    [8,9], [9,10], [10,11], [11,3],
    [12,13], [13,14], [14,15], [15,3],
    [16,17],[17,18],[18,5],
    [19,20],[20,21],[21,5]
] 
bone_color_index = [ 
    0,0,
    3,3,3,3,3,
    1,1,1,1,
    2,2,2,2,
    4,4,4,
    5,5,5
]
# RGB
g_colors = [ 
    [92,94,170], # purple  
    [187,97,166], # pink 
    [109, 192, 91], # green 
    [221,94,86], # red 
    [210, 220, 88], # yellow
    [98,201,211], #blue 
]
g_colors = np.asarray(g_colors, dtype=np.float32) 

joint_color_index = [ 
    0,0,0,
    3,3,3,3,3,
    1,1,1,1,
    2,2,2,2,
    4,4,4,
    5,5,5
]


## pack a list of images into one image. 
def pack_images(imgs): 
    N = len(imgs) 
    h,w = imgs[0].shape[0:2] 
    N_r = int(np.floor(np.sqrt(N))) 
    N_c = N // N_r 
    if N_c * N_r < N: 
        N_c += 1 
    H = N_r * h 
    W = N_c * w
    output = np.zeros([H,W,3], np.uint8)
    for r in range(N_r): 
        for c in range(N_c): 
            k = r * N_c + c 
            output[r*h:(r+1)*h, c*w:(c+1)*w,:] = imgs[k] 
    return output 

def rodrigues_batch(axis):
    # axis : bs * 3
    # return: bs * 3 * 3
    bs = axis.shape[0]
    Imat = torch.eye(3, dtype=axis.dtype, device=axis.device).repeat(bs, 1, 1)  # bs * 3 * 3
    angle = torch.norm(axis, p=2, dim=1, keepdim=True) + 1e-8  # bs * 1
    axes = axis / angle  # bs * 3
    sin = torch.sin(angle).unsqueeze(2)  # bs * 1 * 1
    cos = torch.cos(angle).unsqueeze(2)  # bs * 1 * 1
    L = torch.zeros((bs, 3, 3), dtype=axis.dtype, device=axis.device)
    L[:, 2, 1] = axes[:, 0]
    L[:, 1, 2] = -axes[:, 0]
    L[:, 0, 2] = axes[:, 1]
    L[:, 2, 0] = -axes[:, 1]
    L[:, 1, 0] = axes[:, 2]
    L[:, 0, 1] = -axes[:, 2]
    return Imat + sin * L + (1 - cos) * L.bmm(L)

def Rmat2axis(R):
    # R: bs x 3 x 3
    R = R.view(-1, 3, 3)
    temp = (R - R.permute(0, 2, 1)) / 2
    L = temp[:, [2, 0, 1], [1, 2, 0]]  # bs x 3
    sin = torch.norm(L, dim=1, keepdim=False)  # bs
    L = L / (sin.unsqueeze(-1) + 1e-8)

    temp = (R + R.permute(0, 2, 1)) / 2
    temp = temp - torch.eye((3), dtype=R.dtype, device=R.device)
    temp2 = torch.matmul(L.unsqueeze(-1), L.unsqueeze(1))
    temp2 = temp2 - torch.eye((3), dtype=R.dtype, device=R.device)
    temp = temp[:, 0, 0] + temp[:, 1, 1] + temp[:, 2, 2]
    temp2 = temp2[:, 0, 0] + temp2[:, 1, 1] + temp2[:, 2, 2]
    cos = 1 - temp / (temp2 + 1e-8)  # bs

    sin = torch.clamp(sin, min=-1 + 1e-7, max=1 - 1e-7)
    theta = torch.asin(sin)

    # prevent in-place operation
    theta2 = torch.zeros_like(theta)
    theta2[:] = theta
    idx1 = (cos < 0) & (sin > 0)
    idx2 = (cos < 0) & (sin < 0)
    theta2[idx1] = 3.14159 - theta[idx1]
    theta2[idx2] = -3.14159 - theta[idx2]
    axis = theta2.unsqueeze(-1) * L

    return axis.view(-1, 3)


'''
points: [N,2]
'''
def undist_points_cv2(points, K, coeff, newcameramtx):
    points_cv = points.copy() 
    points_cv = points_cv.reshape([points_cv.shape[0], 1, points_cv.shape[1]])
    new_points_2 = cv2.undistortPoints(points_cv, K, coeff, P=newcameramtx) 
    new_points_2 = new_points_2.squeeze() 
    return new_points_2 # [N,2]

def draw_keypoints(img, proj, bone, is_draw_bone=False): 
    for k in range(proj.shape[0]):
        if math.isnan(proj[k,0]): 
            continue 
        if proj[k,0] == 0 or proj[k,1] == 0: 
            continue 
        p = (int(proj[k,0]), int(proj[k,1]))
        if k < len(joint_color_index): 
            colorid = joint_color_index[k]
        else: 
            colorid = k % colormap.shape[0] 
        cv2.circle(img, p, 9, colormap[colorid].tolist(), -1) 
        # cv2.putText(img, str(k), p, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,128,255))
    if not is_draw_bone: 
        return img 
    for index, b in enumerate(bone): 
        if math.isnan(proj[b[0],0]) or math.isnan(proj[b[1],0]):
            continue 
        p0 = (int(proj[b[0],0]), int(proj[b[0],1]))
        p1 = (int(proj[b[1],0]), int(proj[b[1],1]))
        if p0[0] == 0 or p0[1] == 0 or p1[0] == 1 or p1[1] == 0: 
            continue 
        color = colormap[bone_color_index[index]].tolist()
        cv2.line(img, p0, p1, color, 4)
    return img 