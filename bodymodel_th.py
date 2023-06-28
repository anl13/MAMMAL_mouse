import numpy as np
import pickle
import torch
from torch.nn import Module
import os
from time import time
import json 

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

class BodyModelTorch(Module):
    def __init__(self, model_path_pkl, device=None, 
                    data_type=torch.float32):
        super(BodyModelTorch, self).__init__()
        # self.rotation_type = "axis-angle" # ["axis-angle", "euler"]
        self.rotation_type = "euler"
        self.rotation_func = None 
        if self.rotation_type == "axis-angle": 
            self.rotation_func = self.rodrigues 
        elif self.rotation_type == "euler": 
            self.rotation_func = self.euler2mat 

        self.data_type = data_type
        with open(model_path_pkl, 'rb') as f:
            params = pickle.load(f)
        self.device = device if device is not None else torch.device('cpu')

        self.register_buffer("v_template", to_tensor(params['vertices']))
        self.vertex_num = self.v_template.shape[0]
        self.register_buffer("t_pose_joints", to_tensor(params['t_pose_joints']))
        self.joint_num = self.t_pose_joints.shape[0]
        self.register_buffer("weights", to_tensor(params["skinning_weights"].todense()))
        self.register_buffer("parent", to_tensor(params["parents"], dtype=torch.int64))

        bone_length_mapper = np.loadtxt("mouse_model/mouse_txt/bone_length_mapper.txt", dtype=np.int64).squeeze() 
        self.bone_length_mapper = bone_length_mapper

        self.faces = params["faces_vert"]
        self.faces_reduced_7200 = np.loadtxt("mouse_model/mouse_txt/reduced_face_7200.txt", dtype=np.int64)
        self.reduced_ids = np.loadtxt("mouse_model/mouse_txt/reduced_ids_7200.txt", dtype=np.int64).squeeze().tolist()
        
        self.device = device if device is not None else torch.device('cpu') 
        
        for name in ['weights', 'v_template', 't_pose_joints']:
            _tensor = getattr(self, name)
            print(' Tensor {} shape: '.format(name), _tensor.shape)
            setattr(self, name, _tensor.to(device))

    def read_mapper(self):
        mapper_file = "mouse_model/keypoint22_mapper.json" 
        with open(mapper_file, 'r') as f: 
            mapper = json.load(f) 
        mapper = mapper["mapper"]
        self.mapper = mapper 

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

    @staticmethod
    def with_zeros(x):
        """
        Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

        Parameter:
        ---------
        x: Tensor to be appended.

        Return:
        ------
        Tensor after appending of shape [4,4]

        """
        ones = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=x.dtype
        ).expand(x.shape[0],-1,-1).to(x.device)
        ret = torch.cat((x, ones), dim=1)
        return ret

    @staticmethod
    def pack(x):
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]

        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.

        """
        zeros43 = torch.zeros(
            (x.shape[0], x.shape[1], 4, 3), dtype=x.dtype).to(x.device)
        ret = torch.cat((zeros43, x), dim=3)
        return ret

    def write_obj(self, verts, faces, file_name):
        with open(file_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
        
    '''
        _lR2G: Buildin function, calculating G terms for each vertex.
        bone_lengths: [Batchsize, N_K ]
        lRs: [batchsize, N_K, 3,3]
    '''  
    def _lR2G(self, lRs, J, bone_lengths_core, center_bone_length):
        batch_num = lRs.shape[0]
        results = []    # results correspond to G' terms in original paper.
        results.append(
            self.with_zeros(torch.cat((lRs[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
        )
        for i in range(1, self.joint_num):
            bone_length_id = self.bone_length_mapper[i]
            if i == 1: 
                results.append(
                    torch.matmul(
                    results[self.parent[i]],
                    self.with_zeros(
                        torch.cat(
                        (lRs[:, i], torch.reshape( (J[:, i, :] - J[:, self.parent[i], :]) * center_bone_length, (-1, 3, 1))),
                        # (lRs[:, i], torch.reshape( (J[:, i, :] - J[:, self.parent[i], :]), (-1, 3, 1))),
                        dim=2
                        )
                    )
                    )
                )
            elif bone_length_id < 0: 
                results.append(
                    torch.matmul(
                    results[self.parent[i]],
                    self.with_zeros(
                        torch.cat(
                        (lRs[:, i], torch.reshape( (J[:, i, :] - J[:, self.parent[i], :]), (-1, 3, 1))),
                        dim=2
                        )
                    )
                    )
                )
            else: 
                results.append(
                    torch.matmul(
                    results[self.parent[i]],
                    self.with_zeros(
                        torch.cat(
                        (lRs[:, i], torch.reshape( (J[:, i, :] - J[:, self.parent[i], :]) * bone_lengths_core[:,bone_length_id].view([-1,1]), (-1, 3, 1))),
                        # (lRs[:, i], torch.reshape( (J[:, i, :] - J[:, self.parent[i], :]), (-1, 3, 1))),
                        dim=2
                        )
                    )
                    )
                )
        affine = torch.stack(results, dim=1)
            
        J_final = affine[:,:,:3,3]
        
        deformed_joint = \
            torch.matmul(
            affine,
            torch.reshape(
                torch.cat((J, torch.zeros((batch_num, self.joint_num, 1), dtype=self.data_type).to(self.device)), dim=2),
                (batch_num, self.joint_num, 4, 1)
            )
            ) 
        results = affine - self.pack(deformed_joint)
        return results, J_final
    
    def theta2G(self, thetas, bone_lengths_core, center_bone_length, J):
        batch_num = thetas.shape[0]

        lRs = self.rotation_func(thetas.view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)

        return self._lR2G(lRs, J, bone_lengths_core, center_bone_length)
  
    
    def forward(self, thetas, bone_lengths_core, center_bone_length, trans, scale):
        
        """
            Construct a compute graph that takes in parameters and outputs a tensor as
            model vertices. Face indices are also returned as a numpy ndarray.
                        
            Usage:
            ---------
            meshes, joints = forward(betas, thetas, trans): normal SMPL 
            meshes, joints = forward(betas, thetas, trans, gR=gR): 
                    calling from SMPLModelv3, using gR to cache G terms, ignoring thetas

            Parameters:
            ---------
            thetas: an [N, N_K * 3] tensor indicating child joint rotation
            relative to parent joint. For root joint it's global orientation.
            Represented in a axis-angle format.

            bone_lengths_core: [N, 28], N is batch_size, 28 key bone lengths are mapped to all joints .

            trans: Global translation tensor of shape [N, 3].
            
            
            Return:
            ------
            A 3-D tensor of [N * N_V * 3] for vertices,
            and the corresponding [N * N_K * 3] joint positions.

        """
        batch_num = thetas.shape[0]
        
        J = self.t_pose_joints.tile([batch_num, 1, 1]) 

        bone_lengths_core = torch.sigmoid(bone_lengths_core / 5) * 2 
        G, J_final = self.theta2G(thetas, bone_lengths_core, center_bone_length, J)  # pre-calculate G terms for skinning

        v_posed = self.v_template.tile([batch_num, 1,1])
        
        # (2) Skinning (W)
        T = torch.tensordot(G, self.weights, dims=([1], [0])).permute(0, 3, 1, 2)
        rest_shape_h = torch.cat(
            (v_posed, torch.ones((batch_num, v_posed.shape[1], 1), dtype=self.data_type).to(self.device)), dim=2
        )
        v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
        v = torch.reshape(v, (batch_num, -1, 4))[:, :, :3]

        V_final = v * scale.reshape([batch_num,1,1]) + torch.reshape(trans, (batch_num, 1, 3))
        J_final = J_final * scale.reshape([batch_num,1,1]) + trans.reshape([batch_num, 1, 3])
        self.V_posed = V_final 
        self.J_posed = J_final  
        return V_final, J_final

    def forward_keypoints22(self):
        batch_num = self.V_posed.shape[0] 
        keypoints = torch.zeros((batch_num, 22,3), dtype=self.data_type).to(self.device)
        for k in range(len(self.mapper)):
            map_type = self.mapper[k]["type"]
            keypoint_id = self.mapper[k]["keypoint"]
            if keypoint_id >= 22: 
                continue
            if map_type=="V": 
                keypoints[:,keypoint_id] = self.V_posed[:,self.mapper[k]["ids"]].mean(axis=1)
            elif map_type == "J": 
                keypoints[:,keypoint_id] = self.J_posed[:,self.mapper[k]["ids"]].mean(axis=1)
        return keypoints 

## 2022.08.01: before adding bone_length and bone-based scale 
## For mouse with 140 joints and 14522 vertices
## On windows 10 with cuda 11.3, NVIDIA RTX 2080Ti
## Both are tested without bone_length effect. 
## batchsize     device      avg runtime(ms)
## -----------------------------------------
## 64            cuda        47.73  
## 64            cpu         89.01  
## 32            cuda        43.75  
## 32            cpu         48.77  
## For pig with 62 joints and 11239 vertices 
## On windows 10 with cuda 11.3, NVIDIA RTX 2080Ti
## batchsize     device      avg runtime(ms)
## -----------------------------------------
## 64            cuda        31.53
## 64            cpu         66.01
## 32            cuda        28.66
## 32            cpu         34.39
def test_gpu(data_type=torch.float32):
    device=torch.device('cpu')
    model = BodyModelTorch(
                        device=device,
                        model_path_pkl = 'mouse_model/mouse.pkl',
                        # model_path_pkl = 'H:/examples/PIG_model/pkl_files/PIG_core.pkl',
                        data_type=data_type
                        )
    model.read_mapper()
    pose_size = model.joint_num * 3
    batch_size = 1
    pose = torch.from_numpy((np.random.rand(batch_size, pose_size) - 0.5) * 0)\
            .type(data_type).to(device)
    trans = torch.from_numpy(np.zeros((batch_size, 3))).type(data_type).to(device)
    scale = torch.from_numpy(np.ones([batch_size,1])).type(data_type).to(device)
    bone_lengths = torch.from_numpy(np.zeros([batch_size,28])).type(data_type).to(device)
    center_bone_length = torch.from_numpy(np.ones([batch_size,1])).type(data_type).to(device) 
    s = time() 
    for i in range(1):
        v_posed, j_posed = model(pose, bone_lengths, center_bone_length, trans, scale)
        keypoints = model.forward_keypoints22() 
        print(keypoints)

    print('Avg Time: {}s'.format((time()-s) / 1))
    
  
def test_euler(): 
    '''
    unit test for euler2mat method. 
    '''
    device=torch.device('cuda') 
    data_type = torch.float32 
    model = BodyModelTorch(device=device,
                            model_path_pkl="mouse_model/mouse.pkl", 
                            data_type=data_type) 
    model.read_mapper() 
    
    batch_size = 40 
    pose_size = 21

    a = np.random.rand(batch_size, pose_size) - 0.5
    a = a.reshape([280,1,3])
    pose = torch.from_numpy(a).type(data_type).to(device)
    pose = pose.reshape([280,1,3])
    R = model.euler2mat(pose)
    from scipy.spatial.transform import Rotation 
    max_err = 0 
    for k in range(280): 
        a1 = a[k] 
        R_transfer = Rotation.from_euler("ZYX", a1) 
        mat = R_transfer.as_matrix() 
        mat_th = R[k].detach().cpu().numpy() 

        diff = mat - mat_th
        print("sample {:3d}".format(k), " : ", np.abs(diff).sum())
        if np.abs(diff).sum() > max_err: 
            max_err = np.abs(diff).sum() 
    print("max err: ", max_err)
        

if __name__ == '__main__':
    # test_gpu()
    test_euler()
