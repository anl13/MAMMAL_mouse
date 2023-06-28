"""
This file is used to simulate the skinning process and constraints used in blender. 

By Liang AN, 2022.10.19
"""

import numpy as np 
import pickle 
from scipy.sparse import csc_matrix,csr_matrix,coo_matrix
from scipy.spatial.transform import Rotation 

import torch 
from torch.nn import Module 
import os 
from time import time 
import json 

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

class ArticulationTorch(Module): 
    def __init__(self): 
        super(ArticulationTorch, self).__init__() 
        self.device = torch.device("cuda")
        self.rotation_type = "axis-angle"
        self.rotation_func = None 
        if self.rotation_type == "axis-angle": 
            self.rotation_func = self.rodrigues 
        elif self.rotation_type == "euler": 
            self.rotation_func = self.euler2mat 

        self._read_params() 
        self.tpose_joints_th = self.compute_Tpose()[0] # [joint num, 3]
        # self.init_params(4) 
        self.read_mapper() 
        self.read_reduced()

    def read_mapper(self):
        mapper_file = "mouse_model/keypoint22_mapper.json" 
        with open(mapper_file, 'r') as f: 
            mapper = json.load(f) 
        mapper = mapper["mapper"]
        self.mapper = mapper 

    def read_reduced(self): 
        self.faces_reduced_7200 = np.loadtxt("mouse_model/mouse_txt/reduced_face_7200.txt", dtype=np.int64)
        self.reduced_ids = np.loadtxt("mouse_model/mouse_txt/reduced_ids_7200.txt", dtype=np.int64).squeeze().tolist()
        

    def forward_keypoints22(self):
        batch_num = self.V_final.shape[0] 
        keypoints = torch.zeros((batch_num, 22,3), dtype=torch.float32).to(self.device)
        for k in range(len(self.mapper)):
            map_type = self.mapper[k]["type"]
            keypoint_id = self.mapper[k]["keypoint"]
            if keypoint_id >=22: 
                continue 
            if map_type=="V": 
                keypoints[:,keypoint_id] = self.V_final[:,self.mapper[k]["ids"]].mean(axis=1)
            elif map_type == "J": 
                keypoints[:,keypoint_id] = self.J_final[:,self.mapper[k]["ids"]].mean(axis=1)
        return keypoints 


    def init_params(self, batch_size = 1): 
        self.batch_size = batch_size 
        self._init_rot_vec_th = torch.tensor(self.init_joint_rotvec_np, dtype=torch.float32, device=self.device)
        self.pose_rot_vec = torch.zeros([batch_size, self.jointnum, 3], dtype=torch.float32, device=self.device)
        self.pose_trans = torch.zeros([batch_size, self.jointnum, 3], dtype=torch.float32, device=self.device)
        self.pose_rot_vec += self._init_rot_vec_th 
        self.pose_trans += self._init_joint_trans_th 

        return self.pose_rot_vec, self.pose_trans

    def _read_params(self):
        pklfolder = "mouse_model/mouse_txt/"
        txtfolder="mouse_model/mouse_txt"
        self.vertices_raw_tpose_np = np.loadtxt(txtfolder + "/vertices.txt")

        self.textures_np = np.loadtxt(txtfolder+"/textures.txt")
        self.faces_vert_np = np.loadtxt(txtfolder+"/faces_vert.txt").astype(np.int64)
        self.faces_tex_np = np.loadtxt(txtfolder+"/faces_tex.txt").astype(np.int64)
        with open(pklfolder +"/id_to_names.pkl", 'rb') as f: 
            self.id_to_names = pickle.load(f) # list of names 
        with open(pklfolder +"/names_to_id.pkl", 'rb') as f: 
            self.names_to_id = pickle.load(f) # dict: from name to id 
        with open(pklfolder +"/parents.pkl", 'rb') as f: 
            self.parents = pickle.load(f) 
        with open(pklfolder +"/init_joint_trans.pkl", 'rb') as f: 
            self.init_joint_trans_list = pickle.load(f) 
            self.init_joint_trans_np = np.asarray(self.init_joint_trans_list)
        with open(pklfolder +"/init_joint_rot_mat.pkl", 'rb') as f: 
            self.init_joint_rot_mat_list = pickle.load(f) 
            self.init_joint_rot_mat_np = np.asarray(self.init_joint_rot_mat_list)

        with open(pklfolder + "/init_joint_rotvec.pkl", 'rb') as f: 
            self.init_joint_rotvec_list = pickle.load(f) 
            self.init_joint_rotvec_np = np.asarray(self.init_joint_rotvec_list)
    
        self.jointnum = len(self.id_to_names)
        self.vertexnum = self.vertices_raw_tpose_np.shape[0] 
        self.weights_np = np.zeros((self.vertexnum, self.jointnum))
        _weights = np.loadtxt(txtfolder + "/skinning_weights.txt")
        
        for i in range(_weights.shape[0]):
            jointid = int(_weights[i,0])
            vertexid = int(_weights[i,1])
            value = _weights[i,2]
            self.weights_np[vertexid, jointid] = value 

        # register buffer
        self.register_buffer("v_template_th", to_tensor(self.vertices_raw_tpose_np)) 
        self.register_buffer("weights_th", to_tensor(self.weights_np))
        self.register_buffer("parent_th", to_tensor(self.parents,  dtype=torch.int64))
        self.register_buffer("_init_joint_rot_mat_th", to_tensor(self.init_joint_rot_mat_np))
        self.register_buffer("_init_joint_trans_th", to_tensor(self.init_joint_trans_np))

        for name in ['v_template_th', 'weights_th', 'parent_th', "_init_joint_rot_mat_th", "_init_joint_trans_th"]:
            _tensor = getattr(self, name)
            # print(' Tensor {} shape: '.format(name), _tensor.shape)
            setattr(self, name, _tensor.to(self.device))

        bone_length_mapper = np.loadtxt(txtfolder + "/bone_length_mapper.txt", dtype=np.int64).squeeze() 
        self.bone_length_mapper = bone_length_mapper

    @staticmethod 
    def euler2mat(r):
        """
        turn euler angles (ZYX format) into rotation matrix in batch-ed manner. 
        r: euler rotation [batch-size * angle_num, 1, 3]
        
        return: 
        rotation matrix [batchsize*anglenum, 3,3]
        """ 
        N = r.shape[0] 
        z = r[:,0,0]
        y = r[:,0,1]
        x = r[:,0,2]
        cx = torch.cos(x) 
        sx = torch.sin(x) 
        cy = torch.cos(y) 
        sy = torch.sin(y) 
        cz = torch.cos(z) 
        sz = torch.sin(z) 
        Rx = torch.zeros([N, 3,3],dtype=r.dtype).to(r.device) 
        Rx[:,0,0] = 1 
        Rx[:,1,1] = cx  
        Rx[:,2,2] = cx 
        Rx[:,1,2] = -sx 
        Rx[:,2,1] = sx 
        Ry = torch.zeros([N, 3,3],dtype=r.dtype).to(r.device) 
        Ry[:,0,0] = cy 
        Ry[:,1,1] = 1 
        Ry[:,2,2] = cy
        Ry[:,0,2] = sy 
        Ry[:,2,0] = -sy  
        Rz = torch.zeros([N, 3,3],dtype=r.dtype).to(r.device) 
        Rz[:,0,0] = cz 
        Rz[:,1,1] = cz  
        Rz[:,2,2] = 1
        Rz[:,0,1] = -sz 
        Rz[:,1,0] = sz 

        R1 = torch.einsum("aij,ajk->aik",Rz,Ry) 
        return torch.einsum("aij,ajk->aik",R1,Rx)

    @staticmethod
    def rodrigues(r):
        """
        Rodrigues' rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size * angle_num, 3, 3].

        """
        eps = r.clone().normal_(std=1e-8)
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=r.dtype).to(r.device)
        m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
            -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = (torch.eye(3, dtype=r.dtype).unsqueeze(dim=0) \
                + torch.zeros((theta_dim, 3, 3), dtype=r.dtype)).to(r.device)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    def write_obj(self, file_name):
        with open(file_name, 'w') as fp:
            for v in self.V_final[0]:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces_vert_np + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    ## S is single affine matrix
    ## S.shape: [batchsize, jointnum, 4,4]
    ## chest_deformer: must be a [batchsize, 3, 3] matrix
    def compute_G(self, S, chest_deformer=None): 
        G = []
        for jid in range(self.jointnum): 
            if jid == 0: 
                G.append(S[:, jid])
            else: 
                p = self.parents[jid] 
                # G[:,jid] = torch.einsum("aij,ajk->aik", G[:,p], S[:,jid])
                if jid == 119 and chest_deformer is not None: 
                    M = S[:,jid].clone() 
                    M = M @ chest_deformer
                    G.append(G[p] @ M) 
                else: 
                    G.append(G[p] @ S[:,jid])
        G_stack = torch.stack(G, dim=1)
        return G_stack
        
    ## This function is only called once at the initialization stage. 
    def compute_Tpose(self): 
        single_affine = torch.zeros([1, self.jointnum, 4,4], dtype=torch.float32, device=self.device) 
        single_affine[0,:,0:3,0:3] = self._init_joint_rot_mat_th 
        single_affine[0,:,0:3,3] = self._init_joint_trans_th 
        single_affine[0,:,3,3] = 1 
        G = self.compute_G(single_affine) 
        joints = self.get_joints_from_G(G) 
        return joints 
  
    def get_joints_from_G(self, G): 
        joints = G[:,:,0:3,3]
        return joints 

    def compute_trans(self, joints):
        trans = torch.zeros(joints.shape, dtype=torch.float32, device=self.device)

        for i in range(self.jointnum):
            if i == 0: 
                trans[:,i] = joints[:,i]
            else: 
                p = self.parents[i]
                trans[:,i] = joints[:,i] - joints[:,p]
        return trans 

    def compute_rots(self, G_init, G_pose): 
        G_rel = torch.zeros(G_init.shape, dtype=torch.float32, device=self.device) 
        G_rel = torch.linalg.solve(G_init.transpose(2,3), G_pose.transpose(2,3)).transpose(2,3) 
        R_diff = torch.zeros(G_init.shape, dtype=torch.float32, device=self.device) 
        for i in range(self.jointnum): 
            if i == 0: 
                R_diff[:,i] = G_rel[:,i]
            else: 
                R_diff[:,i] = torch.linalg.solve(G_rel[:,self.parents[i]], G_rel[:,i])
        return R_diff 


    def relative_to_Tpose(self, pose_rot_mat, pose_trans, chest_deformer): 
        assert pose_rot_mat.shape[0] == pose_trans.shape[0]
        batch_size = pose_rot_mat.shape[0] 
        S = torch.zeros([batch_size, self.jointnum, 4, 4], dtype=torch.float32, device=self.device) 
        S[:,:,0:3,0:3] += self._init_joint_rot_mat_th
        S[:,:,0:3, 3] += pose_trans 
        S[:,:,3,3] = 1 
        G = self.compute_G(S) 
        tpose_posed_joints = self.get_joints_from_G(G) 

        S2 = torch.zeros([batch_size, self.jointnum, 4, 4], dtype=torch.float32, device=self.device)
        S2[:,:,0:3,0:3] += self._init_joint_rot_mat_th 
        S2[:,:,0:3,3] += self._init_joint_trans_th 
        S2[:,:,3,3] = 1 
        G2 = self.compute_G(S2) 
        tpose_joints = self.get_joints_from_G(G2) 

        tpose_posed_trans  = self.compute_trans(tpose_posed_joints) 
        tpose_origin_trans = self.compute_trans(tpose_joints) 
        self.diff_trans = tpose_posed_trans - tpose_origin_trans 
   
        G_current_33 = self.compute_G(pose_rot_mat, chest_deformer)

        self.diff_rot_mat = self.compute_rots(G[:,:,0:3,0:3], G_current_33) 

        ## compute deformation G 
        # see T-pose as init pose, apply relative rot and trans towards T-pose 
        # This is used for demonstration
        # return G function of current pose relative to T-pose
        # used for skinning from the t-pose mesh. 
        # because we have only the t-pose vertices
        S3 = torch.zeros([batch_size, self.jointnum, 4,4], dtype=torch.float32, device=self.device)
        for i in range(self.jointnum): 
            if i == 0: 
                S3[:,i,0:3,3] = self.diff_trans[:,i]
                S3[:,i,0:3,3] += self.tpose_joints_th[i]
            else: 
                S3[:,i,0:3,3] = self.diff_trans[:,i] 
                S3[:,i,0:3,3] += (self.tpose_joints_th[i] - self.tpose_joints_th[self.parents[i]])
            S3[:,i,0:3,0:3] = self.diff_rot_mat[:,i]
            S3[:,i,3,3] = 1 
        G_out = self.compute_G(S3) 

        return G_out 

    def skinning(self, G): 
        # Gn = torch.zeros(G.shape, dtype=torch.float32, device=self.device) 
        Gn = G 
        batch_size = G.shape[0] 
        tpose_joints_th_tile = self.tpose_joints_th.tile([batch_size, 1, 1]) 

        for i in range(self.jointnum): 
            Gn[:,i,0:3,3] = G[:,i,0:3,3] - torch.einsum("aij,aj->ai", G[:,i,0:3,0:3], tpose_joints_th_tile[:,i,:])
            
        # Ga = torch.tensordot(Gn, self.weights_th.T, dims=([1],[0])).permute(0,3,1,2) 
        Ga = torch.einsum("ijkm,nj->inkm", Gn, self.weights_th) 

        v_posed = self.v_template_th.tile([batch_size, 1,1])

        rest_shape_h = torch.cat(
            (v_posed, torch.ones((batch_size, v_posed.shape[1], 1), dtype=torch.float32, device=self.device)), dim=2
        )
        v = torch.matmul(Ga, torch.reshape(rest_shape_h, (batch_size, -1, 4, 1)))
        v = torch.reshape(v, (batch_size, -1, 4))[:, :, :3]

        return v 

    def forward(self, 
            thetas, 
            bone_lengths_core, 
            R, T, s, 
            chest_deformer            
        ):
        '''
        thetas: [batchsize, 140, 3], axis-angle or euler angles 
        bone_lengths_core: [batchsize, 28]
        center_bone_length: [batchsize, 1]
        global_R: [batchsize, 3], axis-angle 
        global_T: [batchsize, 3], translation vector 
        global_s: [batchsize, 1], global scale 
        chest_deformer: [batchsize, 1], y-axis deformation of chest scale deformation 
        belly_stretch_deformer
        '''
        batch_size = thetas.shape[0]
        
        pose_trans = self._init_joint_trans_th.tile([batch_size, 1, 1])
        pose_trans[:,1] *= 0 # make center bone length = 0
        bone_lengths_core = torch.sigmoid(bone_lengths_core) + 0.5
        for i in range(self.jointnum): 
            bone_length_id = self.bone_length_mapper[i] - 1
            if bone_length_id < 0: 
                continue 
            else: 
                pose_trans[:,i] *= bone_lengths_core[:,bone_length_id]

        pose_rot_mat = self.rotation_func(thetas.view(-1, 1, 3)).reshape(batch_size, -1, 3, 3)
        chest_deformer = torch.sigmoid(chest_deformer) * 2 + 0.2 # [0.2, 2.2]

        chest_deformer1 = torch.zeros([batch_size, 3,3], dtype=torch.float32, device=self.device) 
        chest_deformer1[:,0,0] = torch.sqrt(1/chest_deformer)
        chest_deformer1[:,1,1] = chest_deformer
        chest_deformer1[:,2,2] = torch.sqrt(1/chest_deformer)

        # pose_rot_mat[:,119,:,0] *= 10 
        # pose_rot_mat[:,119,:,1] *= 0.01 
        # pose_rot_mat[:,119,:,2] *= 10 

        G_out = self.relative_to_Tpose(pose_rot_mat, pose_trans, chest_deformer1)
        self.J_final = self.get_joints_from_G(G_out).clone() 
        self.V_final = self.skinning(G_out) 
        global_R = self.rotation_func(R.view(-1,1,3)).reshape([batch_size, 3,3])
        self.J_final = self.J_final @ global_R
        self.V_final = self.V_final @ global_R
        self.J_final = (self.J_final * s.view(batch_size, 1, 1)) + T.view(batch_size, 1, 3)
        self.V_final = (self.V_final * s.view(batch_size, 1, 1)) + T.view(batch_size, 1, 3)
        return self.V_final, self.J_final

if __name__ == "__main__":
    A = ArticulationTorch() 

    with open("mouse_model/mouse_txt/init_joint_rotvec.pkl", 'rb') as f: 
        rotvec = pickle.load(f) 
        rotvec = np.asarray(rotvec)

    thetas = torch.from_numpy(rotvec).type(torch.float32).to(A.device)
    
    batch_size = 1 
    thetas = thetas.tile([batch_size,1,1])
    bone_lengths_core = torch.zeros([batch_size, 19], dtype=torch.float32, device=A.device) 
    R = torch.zeros([batch_size, 3], dtype=torch.float32, device=A.device) 
    T = torch.zeros([batch_size, 3], dtype=torch.float32, device=A.device) 
    s = torch.ones([batch_size, 1], dtype=torch.float32, device=A.device) 
    chest_deformer = torch.zeros([batch_size, 1], dtype=torch.float32, device=A.device) + 0.01
    # chest_deformer = torch.randn([batch_size,3,3], dtype=torch.float32, device=A.device)

    thetas.requires_grad_(True) 
    chest_deformer.requires_grad_(True) 
    bone_lengths_core.requires_grad_(True) 
    R.requires_grad_(True) 
    T.requires_grad_(True) 
    s.requires_grad_(True)
    V, J = A.forward(thetas, bone_lengths_core, R, T, s, chest_deformer) 
    A.write_obj("tmp4/test.obj")

    ## Test backward: 
    V_target = torch.zeros([batch_size, A.vertexnum, 3], dtype=torch.float32, device=A.device) 
    J_target = torch.zeros([batch_size, A.jointnum, 3], dtype=torch.float32, device=A.device) 
    loss = torch.mean(torch.norm(V_target-V, dim=-1))
    loss.backward()
    from IPython import embed; embed()
    exit() 
