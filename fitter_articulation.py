import numpy as np 
import math 
import torch 
import os 
import pickle 
from time import time 
import json 
from tqdm import tqdm 

import pyrender 
from pyrender.constants import RenderFlags 
import cv2 

import trimesh 
import torch.nn as nn 
import torch.functional as F
from articulation_th import ArticulationTorch 
from bodymodel_th import BodyModelTorch 
from data_seaker_video_new import DataSeakerDet
import copy 
from utils import *
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt 

from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex,
    HardFlatShader,
    HardGouraudShader,
    AmbientLights,
    SoftSilhouetteShader
)
from pytorch3d.structures import Meshes
from torch.utils.tensorboard import SummaryWriter 
from pytorch3d.utils import cameras_from_opencv_projection

WITH_RENDER = True  
KEYPOINT_NUM = 22
RENDER_CAMERAS = [0,1,2,3,4,5]

class MouseFitter():
    def __init__(self):
        mouse_path = "mouse_model/mouse.pkl"
        img_size = (1024, 1152) #H,W
        self.img_size = img_size
        self.device = torch.device('cuda')
        self.bodymodel = ArticulationTorch() 

        if WITH_RENDER: 
            self.renderer = pyrender.OffscreenRenderer(viewport_width=img_size[1], viewport_height=img_size[0])
        self.reg_weights = np.loadtxt("mouse_model/reg_weights.txt").squeeze()
        self.keypoint_weight = np.ones(KEYPOINT_NUM) 
        self.keypoint_weight[4] = 0.4
        self.keypoint_weight[11] = 0.9
        self.keypoint_weight[15] = 0.9
        self.keypoint_weight[5] = 2 
        self.keypoint_weight[6] = 1.5
        self.keypoint_weight[7] = 1.5
        self.keypoint_weight = torch.from_numpy(self.keypoint_weight).reshape([1,-1,1]).to(self.device)
        
        bone_weight = np.ones(20) 
        bone_weight[11] = 1
        bone_weight[19] = 1
        self.bone_weight = torch.from_numpy(bone_weight).reshape([1,-1,1]).to(self.device) 
        self.data_loader = None 

        self.result_folder = ""

        self.term_weights = { 
            "theta" : 3,
            "3d": 2.5, 
            "2d": 0.2,
            "bone": 0.5,
            "scale": 0.5,
            "mask": 10,
            "chest_deformer": 0.1,
            "stretch": 1,
            "temp": 0.25, 
            "temp_d": 0.2
        }

        self.losses = {}

        ## init differentiable renderer
        sigma = 3e-5
        raster_settings_soft = RasterizationSettings(
            image_size=img_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            # blur_radius=0.01,
            faces_per_pixel=50,
            # bin_size = 56, 
            # max_faces_per_bin= 16
        )
        self.renderer_mask = MeshRenderer( 
            rasterizer=MeshRasterizer(raster_settings=raster_settings_soft),
            shader = SoftSilhouetteShader() 
        )
        faces_np = self.bodymodel.faces_vert_np.astype(np.int64) 
        self.faces_th = torch.from_numpy(faces_np).to(self.device).unsqueeze(0) 
        self.faces_th_reduced = torch.from_numpy(self.bodymodel.faces_reduced_7200).to(self.device).unsqueeze(0)

        self.mask_loss_func = torch.nn.MSELoss()

        ## data loader 
        self.cam_dict = [] 
        self.imgs = [] 
        self.id = None 
        
        self.last_params = None 
        self.V_last = None 
        self.J_last = None 

    def set_cameras_dannce(self, cams):
        self.camN = len(cams)
        self.cam_dict = cams 
        self.cams_th = [] 
        self.Rs = [] 
        self.Ks = [] 
        self.Ts = [] 
        for cam in cams: 
            R = np.expand_dims(cam['R'].T, 0).astype(np.float32)
            K = np.expand_dims(cam['K'].T, 0).astype(np.float32) 
            T = cam['T'].astype(np.float32) 
            img_size_np = np.expand_dims(np.asarray(self.img_size), 0).astype(np.float32)
            cam_th = self.build_opencv_camera(R, T, K, img_size_np)
            self.cams_th.append(cam_th)       
            self.Rs.append(torch.from_numpy(R).to(self.device))
            self.Ts.append(torch.from_numpy(T).to(self.device)) 
            self.Ks.append(torch.from_numpy(K).to(self.device)) 

    # build camera from OpenCV camera format 
    # R: [N, 3,3], np.ndarray
    # T: [N, 3], np.ndarray
    # K: [N,3,3], np.ndarray
    # imgsize:[N, 2 ], np.ndarray, h,w
    def build_opencv_camera(self, R, T, K, imgsize): 
        R_tensor = torch.from_numpy(R).to(self.device) 
        tvec_tensor = torch.from_numpy(T).to(self.device) 
        K_tensor = torch.from_numpy(K).to(self.device) 
        imgsize_tensor = torch.from_numpy(imgsize).to(self.device)
        return cameras_from_opencv_projection(R=R_tensor,
            tvec=tvec_tensor, camera_matrix=K_tensor, image_size=imgsize_tensor)

    def init_params(self, batch_size):
        body_param = { 
            "thetas": np.tile(self.bodymodel.init_joint_rotvec_np, [batch_size, 1, 1]),
            "trans": np.zeros([batch_size,3]), 
            "scale": np.ones([batch_size,1]) * 115,
            "rotation": np.zeros([batch_size, 3]), 
            "bone_lengths": np.zeros([batch_size, 20]),
            "chest_deformer": np.zeros([batch_size, 1])
        }

        self.init_thetas = torch.tensor(body_param["thetas"], dtype=torch.float32, device=self.device)
        
        return body_param 

    def calc_2d_keypoint_loss(self, J3d, x2): 
        loss = 0 
        for camid in range(self.camN): 
            J2d = (J3d@self.Rs[camid].transpose(1,2) + self.Ts[camid]) @ self.Ks[camid].transpose(1,2)
            J2d = J2d / J2d[:,:,2:]
            J2d = J2d[:,:,0:2]

            loss += torch.mean(torch.norm((J2d - x2[:,camid,:,0:2]) * x2[:,camid,:,2:] * self.keypoint_weight, dim=-1) ) 
        return loss     

    def calc_3d_loss(self, x1, x2): 
        res = (x1 - x2) * self.keypoint_weight
        loss = torch.mean(torch.norm(res, dim=-1))
        return loss 

    ## both J_prev and J_curr are embed joints (140, 3)
    def calc_deformer_end_temporal_loss(self, J_prev, J_curr): 
        res = J_prev[:,[50,120]] - J_curr[:,[50,120]]
        loss  = torch.mean(torch.norm(res, dim=-1)) * 5 
        return loss 

    def calc_scale_loss(self, scale, target): 
        loss = torch.mean(torch.norm(scale-target, dim=-1))
        return loss 

    def calc_chest_deformer_loss(self, chest_deformer): 
        loss = torch.mean(torch.norm(chest_deformer, dim=-1))
        return loss 

    def calc_bone_length_constraint(self, bone_lengths): 
        loss = torch.norm(bone_lengths * self.bone_weight)
        return loss 

    def calc_temporal_term(self, params, V, J): 
        loss = 0
        for k, v in params.items():
            loss = loss + torch.mean(torch.norm(params[k] - self.last_params[k], dim=-1))
        loss += torch.mean(torch.norm(V-self.V_last, dim=-1))
        loss += torch.mean(torch.norm(J-self.J_last, dim=-1)) 
        return loss  

    def calc_stretch_to_constraints(self, joints): 
        dist1 = torch.mean(torch.norm(joints[:,50] - joints[:,121], dim=-1)) 
        dist2 = torch.mean(torch.norm(joints[:,65] - joints[:,123], dim=-1)) 
        dist3 = torch.mean(torch.norm(joints[:,72] - joints[:,134], dim=-1)) 
        dist4 = torch.mean(torch.norm(joints[:,97] - joints[:,134], dim=-1)) 
        return dist1 + dist2 + dist3 + dist4

    ## theta regularization. 
    def calc_theta_loss(self, thetas):
        weights = torch.from_numpy(self.reg_weights).reshape([1,-1,1]).type(torch.float32).to(self.device)
        loss_theta = torch.norm( (thetas - self.init_thetas) * weights)
        return loss_theta 

    def set_previous_frame(self, previous_params): 
        self.last_params = {} 
        for k,v in previous_params.items(): 
            self.last_params.update({
                k: v.detach().to(self.device) 
            })
        self.V_last, self.J_last = self.bodymodel.forward(
            self.last_params["thetas"], 
            self.last_params["bone_lengths"],
            self.last_params["rotation"],
            self.last_params["trans"],
            self.last_params["scale"], 
            self.last_params["chest_deformer"]
        )

    def gen_closure(self, optimizer, body_param, target):
        def closure():
            optimizer.zero_grad() 
            V,J = self.bodymodel.forward(body_param["thetas"], body_param["bone_lengths"], \
                body_param["rotation"], body_param["trans"], body_param["scale"], body_param["chest_deformer"])

            keypoints = self.bodymodel.forward_keypoints22() 
            # loss_3d = self.calc_3d_loss(keypoints, target["target_3d"])
            loss_2d = self.calc_2d_keypoint_loss(keypoints, target["target_2d"])
            
            loss_theta = self.calc_theta_loss(body_param["thetas"])

            loss_bone_length = self.calc_bone_length_constraint(body_param["bone_lengths"])

            loss_scale = self.calc_scale_loss(body_param["scale"], 115)

            loss_chest_deformer = self.calc_chest_deformer_loss(body_param["chest_deformer"])

            loss_stretch_to_constraints = self.calc_stretch_to_constraints(J) 


            loss_temp = 0 
            loss_deformer_temp = 0
            if self.last_params is not None: 
                loss_temp = self.calc_temporal_term(body_param, V, J)
                loss_deformer_temp = self.calc_deformer_end_temporal_loss(J_prev=self.J_last, J_curr = J)


            ## mask loss
            V_reduced = V[:,self.bodymodel.reduced_ids,:]
            mesh = Meshes(V_reduced, self.faces_th_reduced)
            loss_mask = torch.tensor(0.0, device=self.device)
            if self.term_weights["mask"] > 0: 
                for k in RENDER_CAMERAS:
                    mask = self.renderer_mask(mesh, cameras = self.cams_th[k])[...,-1]
                    loss_mask += self.mask_loss_func(mask, target["mask"+str(k)]) 

            self.losses.update({
                "theta": round(float(loss_theta.detach().cpu().numpy()), 2),
                "2d": round(float(loss_2d.detach().cpu().numpy()), 2), 
                "bone": round(float(loss_bone_length.detach().cpu().numpy()), 2), 
                "scale" : round(float(loss_scale.detach().cpu().numpy()), 2),
                "mask": round(float(loss_mask.detach().cpu().numpy()), 2),
                "chest_d": round(float(loss_chest_deformer.detach().cpu().numpy()), ), 
                "stretch": round(float(loss_stretch_to_constraints.detach().cpu().numpy()), 2)
            })
            if loss_temp > 0: 
                self.losses.update(
                    {
                        "temp": round(float(loss_temp.detach().cpu().numpy()), 2), 
                        "temp_d": round(float(loss_deformer_temp.detach().cpu().numpy()), 2)
                    }
                )

            loss_v = loss_2d * self.term_weights["2d"] \
                + loss_theta * self.term_weights["theta"] \
                + loss_bone_length * self.term_weights["bone"] \
                + loss_scale * self.term_weights["scale"] \
                + loss_mask * self.term_weights["mask"] \
                + loss_chest_deformer * self.term_weights["chest_deformer"] \
                + loss_stretch_to_constraints * self.term_weights["stretch"] \
                + loss_temp * self.term_weights["temp"] \
                + loss_deformer_temp * self.term_weights["temp_d"]

            loss_v.backward() 
            return loss_v 
        return closure

    def solve_step0(self, params, target, max_iters):
        ## step1: optimize skeleton
        tolerate = 1e-2
        optimizer = torch.optim.LBFGS(params.values(), line_search_fn="strong_wolfe")
        # optimizer = torch.optim.Adam(params.values(), lr=0.001)
        closure = self.gen_closure(
                    optimizer, params, target
                )

        loss_prev = float('inf')
        self.keypoint_weight[:,16:19,:] = 1
        self.keypoint_weight[:,19:22,:] = 1
        self.term_weights["mask"] = 0
        self.term_weights["stretch"] = 0
        params["chest_deformer"].requires_grad_(False)    
        params["thetas"].requires_grad_(False)
        params["bone_lengths"].requires_grad_(False)
        for i in range(max_iters): 
            loss = optimizer.step(closure).item() 
            print(self.losses) 
            if abs(loss-loss_prev) < tolerate: 
                break 
            else: 
                print('iter ' + str(i) + ': ' + '%.2f'%loss + "  diff: " + "%.2f"%(loss-loss_prev))
            loss_prev = loss 
            if WITH_RENDER: 
                imgs = self.imgs.copy() 
                self.render(params, imgs, RENDER_CAMERAS, 0, self.result_folder + "/render/debug/fitting_{}_global_iter_{:05d}.png".format(self.id, i), self.cam_dict)
        params["chest_deformer"].requires_grad_(True)    
        params["thetas"].requires_grad_(True)
        params["bone_lengths"].requires_grad_(True)
        return params

    def solve_step1(self, params, target, max_iters):
        ## step1: optimize skeleton
        tolerate = 1e-4
        optimizer = torch.optim.LBFGS(params.values(), line_search_fn="strong_wolfe")
        # optimizer = torch.optim.Adam(params.values(), lr=0.001)
        closure = self.gen_closure(
                    optimizer, params, target
                )

        loss_prev = float('inf')
        self.keypoint_weight[:,16:19,:] = 1
        self.keypoint_weight[:,19:22,:] = 1
        self.term_weights["mask"] = 0
        self.term_weights["stretch"] = 0
        params["chest_deformer"].requires_grad_(False)    
        for i in range(max_iters): 
            loss = optimizer.step(closure).item() 
            print(self.losses) 
            if abs(loss-loss_prev) < tolerate: 
                break 
            else: 
                print('iter ' + str(i) + ': ' + '%.2f'%loss + "  diff: " + "%.2f"%(loss-loss_prev))

            loss_prev = loss 
            if self.id == 0 and WITH_RENDER: 
                imgs = self.imgs.copy() 
                self.render(params, imgs, RENDER_CAMERAS, 0, self.result_folder + "/render/debug/fitting_{}_debug_iter_{:05d}.png".format(self.id, i), self.cam_dict)
                

        if WITH_RENDER:
            imgs = self.imgs.copy() 
            self.render(params, imgs, RENDER_CAMERAS, 0, self.result_folder + "/render/fitting_{}.png".format(self.id), self.cam_dict)
            self.draw_keypoints_compare(params, imgs, RENDER_CAMERAS, 0, self.result_folder + "/fitting_keypoints_{}.png".format(self.id), self.cam_dict)
        with open(self.result_folder + "/params/param{}.pkl".format(self.id), 'wb') as f: 
            pickle.dump(params,f) 
        params["chest_deformer"].requires_grad_(True) 
        return params

    def solve_step2(self, params, target, max_iters): 
        ## step2: optim with mask 
        ## enlarge foot keypoint weight because mask on foot are bad. 
        tolerate = 1e-4
        optimizer = torch.optim.LBFGS(params.values(), line_search_fn="strong_wolfe")
        # optimizer = torch.optim.Adam(params.values(), lr=0.001)
        closure = self.gen_closure(
                    optimizer, params, target
                )

        self.keypoint_weight[:,16:19,:] = 10
        self.keypoint_weight[:,19:22,:] = 10
        self.term_weights["mask"] = 3000
        self.term_weights["stretch"] = 0
        loss_prev = float('inf')
        optimizer.zero_grad() 
        for i in range(max_iters): 
            loss = optimizer.step(closure).item()  
            print(self.losses) 
            if abs(loss-loss_prev) < tolerate: 
                break 
            else: 
                print('iter ' + str(i) + ': ' + '%.2f'%loss + "  diff: " + "%.2f"%(loss-loss_prev))

            loss_prev = loss 
        if WITH_RENDER: 
            imgs = self.imgs.copy() 
            self.render(params, imgs, RENDER_CAMERAS, 0, self.result_folder+"/render/fitting_{}_sil.png".format(self.id), self.cam_dict)
            # self.draw_keypoints_compare(params, imgs, RENDER_CAMERAS, 0, self.result_folder + "/fitting_keypoints_{}_sil.png".format(self.id), self.cam_dict)
        with open(self.result_folder + "/params/param{}_sil.pkl".format(self.id), 'wb') as f: 
            pickle.dump(params,f) 
        self.result = params 
        return params 

    def render(self, result, imgs, views, batch_id, filename, cams_dict): 
        V,J = self.bodymodel.forward(result["thetas"], result["bone_lengths"], \
            result["rotation"], result["trans"] / 1000, result["scale"] / 1000, result["chest_deformer"])
        vertices = V[batch_id].detach().cpu().numpy() 
        faces = self.bodymodel.faces_vert_np
        scene = pyrender.Scene() #ambient_light=np.ones(3), bg_color=np.array([0.5, 0.5, 0.5]))
        # light_node = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0))
        light_node = scene.add(pyrender.PointLight(color=np.ones(3), intensity=0.2))
        scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_colors=np.array([0.8, 0.6, 0.4]))))
        color_maps = []
        for view in views:
            cam_param = cams_dict[view]
            K, R, T = cam_param['K'].T, cam_param['R'].T, cam_param['T'].T / 1000
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = R.T
            camera_pose[:3, 3:4] = np.dot(-R.T, T)
            camera_pose[:, 1:3] = -camera_pose[:, 1:3]
            camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
            cam_node = scene.add(camera, name='cam', pose=camera_pose)
            light_node._matrix = camera_pose
            color, _ = self.renderer.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
            scene.remove_node(cam_node)
            img_i = imgs[view]
            color = copy.deepcopy(color)
            background_mask = color[:, :, :] == 255
            color[background_mask] = img_i[background_mask]
            color_maps.append(color) 
        output = pack_images(color_maps) 
        if filename is not None: 
            cv2.imwrite(filename, output)
        return output
        

    def draw_keypoints_compare(self, result, imgs, views, batch_id, filename, cams_dict): 
        myimages = imgs.copy()
        V,J = self.bodymodel.forward(result["thetas"], result["bone_lengths"], \
            result["rotation"], result["trans"] / 1000, result["scale"] / 1000, result["chest_deformer"])
        keypoints = self.bodymodel.forward_keypoints22() 
        joints = keypoints[batch_id].detach().cpu().numpy() 
        all_drawn_images = [] 
        for camid in views: 
            cam = cams_dict[camid] 
            data2d = (joints@cam['R'] + cam["T"] / 1000)@cam["K"] 
            data2d = data2d[:,0:2] / data2d[:,2:] 
            img_drawn = draw_keypoints(myimages[camid], data2d, bones, is_draw_bone=True)
            all_drawn_images.append(img_drawn) 
        outputimg = pack_images(all_drawn_images) 
        cv2.imwrite(filename, outputimg)

import argparse 
def optim_single():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--start", type=int, default=0) 
    parser.add_argument("--end", type=int, default=1) 
    parser.add_argument("--interval", type=int, default=1) 
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--resume", type=bool, default=False)
    args = parser.parse_args() 
    data_name = "markerless_mouse_1"
    data_loader = DataSeakerDet()
    device = torch.device('cuda')

    fitter = MouseFitter()
    fitter.set_cameras_dannce(data_loader.cams_dict_out)
    camN = len(data_loader.cams_dict_out)
    fitter.result_folder = "mouse_fitting_result/results/"
    os.makedirs(fitter.result_folder, exist_ok=True)
    subfolders = ["params", "render", "render/debug/"]
    for subfolder in subfolders: 
        os.makedirs(os.path.join(fitter.result_folder, subfolder), exist_ok=True)

    print("camN: ", camN)
    targets = np.arange(args.start, args.end, args.interval).tolist() 

    start = targets[0]
    for index in targets:
        print("process ... ", index) 
        labels = data_loader.fetch(index, with_img = WITH_RENDER) 
        fitter.id = index
        fitter.imgs = labels["imgs"] 

        if index == start: 
            if args.resume: 
                with open(os.path.join(fitter.result_folder, "params/param{}_sil.pkl".format(index-1)), 'rb') as f: 
                    init_params = pickle.load(f)
                fitter.set_previous_frame(init_params) 
                params = init_params 
            else: 
                init_params = fitter.init_params(1) 
                params = {k: torch.tensor(v, dtype=torch.float32, device=device).requires_grad_(True) for k, v in init_params.items()}
        target = {
            'target_2d': torch.FloatTensor(labels["label2d"]).reshape([-1,camN,KEYPOINT_NUM,3]).to(device)
        }
        for viewid in range(camN): 
            target.update({ 
                "mask{}".format(viewid): torch.from_numpy(np.expand_dims(labels["bgs"][viewid],axis=0)).to(device) # should be [bs, ]
            })
        
        if index == start: 
            params = fitter.solve_step0(params = params, target=target, max_iters = 10)
            params = fitter.solve_step1(params = params, target=target, max_iters = 100)
            params = fitter.solve_step2(params = params, target=target, max_iters = 30)
        else: 
            params = fitter.solve_step1(params = params, target=target, max_iters = 10)
            params = fitter.solve_step2(params = params, target=target, max_iters = 30)
        fitter.set_previous_frame(params) 

if __name__ == "__main__":
    optim_single()
 