import numpy as np 
import os 
from scipy.spatial.transform import Rotation
import json 
import pickle 

class BodyModelNumpy(object):
    def __init__(self, model_folder = "mouse_model/mouse_txt/"):
        self.euler_root = False 
        self.model_folder = model_folder
        self.vertices, self.parents, self.joints, self.weights, self.faces, self.faces_tex, self.textures = \
            self.readmodel(self.model_folder)
        self.joint_num = self.joints.shape[0]

        self.translation = np.zeros(3, dtype=np.float32) 
        self.poseparam = np.zeros([self.joint_num, 3], dtype=np.float32)
        self.scaleparam = np.zeros([self.joint_num, 3], dtype=np.float32) ## some joints contain scale effect which use scale change to deform vertices
        self.scale = 1

        self.posed_joints = self.joints.copy() 
        self.posed_vertices = self.vertices.copy() 

    def readstate(self,filename):
        states = np.loadtxt(filename)
        translation = states[0:3]
        scale = states[-1]
        poseparam = states[3:-1].reshape([-1,3])
        
        self.translation = translation
        self.poseparam = poseparam
        self.scale = scale 
        return translation, poseparam, scale

    def read_mapper(self):
        mapper_file = "mouse_model/keypoint22_mapper.json" 
        with open(mapper_file, 'r') as f: 
            mapper = json.load(f) 
        mapper = mapper["mapper"]
        self.mapper = mapper 

    def readmodel(self, model_folder):
        vertices_np = np.loadtxt(os.path.join(model_folder, "vertices.txt"))
        parents_np = np.loadtxt(os.path.join(model_folder, "parents.txt"), dtype=np.int32).squeeze()
        joints_np = np.loadtxt(os.path.join(model_folder, "t_pose_joints.txt"))
        weights = np.zeros((vertices_np.shape[0], parents_np.shape[0]))
        _weights = np.loadtxt(os.path.join(model_folder, "skinning_weights.txt"))
        for i in range(_weights.shape[0]):
            jointid = int(_weights[i,0])
            vertexid = int(_weights[i,1])
            value = _weights[i,2]
            weights[vertexid, jointid] = value 
        faces_vert = np.loadtxt(os.path.join(model_folder, "faces_vert.txt"), dtype=np.int32)
        faces_tex = np.loadtxt(os.path.join(model_folder, "faces_tex.txt"), dtype=np.int32)
        textures = np.loadtxt(os.path.join(model_folder, "textures.txt"))
        
        return vertices_np, parents_np, joints_np, weights, faces_vert, faces_tex, textures

    def poseparam2Rot(self, poseparam):
        Rot = np.zeros((poseparam.shape[0], 3, 3), dtype=np.float32)
        if self.euler_root: 
            r_tmp1 = Rotation.from_euler('ZYX', poseparam[0], degrees=False)
            Rot[0] = r_tmp1.as_matrix()
            r_tmp2 = Rotation.from_rotvec(poseparam[1:])
            Rot[1:] = r_tmp2.as_matrix()
        else: 
            r_tmp2 = Rotation.from_rotvec(poseparam[0:])
            Rot[0:] = r_tmp2.as_matrix()
        return Rot

    def write_obj(self, filename):
        with open(filename, 'w') as fp:
            for v in self.posed_vertices:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def joint_Rot(self, Rot, bone_lengths):
        skinmat = np.repeat(np.eye(4, dtype=np.float32).reshape(1, 4, 4), repeats=self.joints.shape[0], axis=0)
        skinmat[0, :3, :3] = Rot[0]
        skinmat[0, :3, 3] = self.joints[0]
        
        skinmat_global = skinmat.copy() 
        for jind in range(1, self.joints.shape[0]):
            skinmat[jind, :3, :3] = Rot[jind]
            skinmat[jind, :3, 3] = (self.joints[jind] - self.joints[self.parents[jind]]) * bone_lengths[jind]
            if jind == 0: 
                skinmat_global[jind] = skinmat[jind]
            else: 
                skinmat_global[jind] = np.matmul(skinmat_global[self.parents[jind]], skinmat[jind])

        joints_final = skinmat_global[:, :3, 3].copy()
        joints_deformed = np.zeros((self.joints.shape[0], 4), dtype=np.float32)
        for jind in range(self.joints.shape[0]):
            joints_deformed[jind, :3] = np.matmul(skinmat_global[jind, :3, :3], self.joints[jind])
        skinmat_normed = skinmat_global.copy() 
        skinmat_normed[:, :, 3] = skinmat_normed[:, :, 3] - joints_deformed
        
        return skinmat_normed[:, :3, :], joints_final

    def regress_verts(self, skinmat_normed):
        vertsmat = np.tensordot(self.weights, skinmat_normed, axes=([1], [0]))
        verts_final = np.zeros((self.vertices.shape[0], 3), dtype=np.float32)
        for vind in range(self.vertices.shape[0]):
            verts_final[vind] = np.matmul(vertsmat[vind, :, :3], self.vertices[vind]) + vertsmat[vind, :, 3]
        
        return verts_final

    ### lbs process
    ## pose: [N_K, 3], N_K is joint number 
    ## bone_lengths: [N_K, 1], N_K is joint number. Always set 1 at root. 
    ## trans: [3] global translation 
    ## scale: [1] global scale. 
    def forward(self, pose, bone_lengths, trans=np.zeros(3, dtype=np.float32), scale=1):
        rot = self.poseparam2Rot(pose)
        skinmat_normed, joints_final = self.joint_Rot(rot, bone_lengths)
        self.posed_joints = joints_final * scale + trans 
        verts = self.regress_verts(skinmat_normed) 
        self.posed_vertices = verts * scale + trans 
        return self.posed_vertices, self.posed_joints

    ### regress keypoints from mapper 
    def forward_keypoints22(self):
        keypoints = np.zeros([22,3])
        for k in range(22):
            map_type = self.mapper[k]["type"]
            if map_type=="V": 
                keypoints[k] = self.posed_vertices[self.mapper[k]["ids"]].mean(axis=1)
            elif map_type == "J": 
                keypoints[k] = self.posed_joints[self.mapper[k]["ids"]].mean(axis=1)
        return keypoints 

    # def regress_keypoints(self):
            # [target joint, type, source index, weight]
        # self.optimize_pair = [
        #     [ 0, 1, 10895, 1 ], # nose
        #     [ 1, 1, 938, 1 ], # left eye
        #     [ 2, 1, 6053, 1 ], # right eye
        #     [ 3, 1, 1368, 1 ], # left ear
        #     [ 4, 1, 6600, 1 ], # right ear
        #     [ 5, 0, 15, 1 ], # left shouder
        #     [ 6, 0, 7, 1 ], # right shoulder
        #     [ 7, 0, 16, 1 ], # left elbow
        #     [ 8, 0, 8, 1 ], # right elbow
        #     [ 9, 0, 17, 1 ], # left paw
        #     [ 10, 0, 9, 1 ], # right paw
        #     [ 11, 0, 56, 1 ], # left hip
        #     [ 12, 0, 40, 1 ], # right hip
        #     [ 13, 0, 57, 1 ], # left knee
        #     [ 14, 0, 41, 1 ], # right knee
        #     [ 15, 0, 58, 1 ], # left foot
        #     [ 16, 0, 42, 1 ], # right foot
        #     [ 17, -1, 0, 0], # neck(not use)
        #     [ 18, 1, 7903, 1 ], # tail 
        #     [19, -1, 0,0], #wither (not use) 
        #     [ 20, 0, 2, 1 ], # center
        #     [21, -1, 0,0], # tail middle (not use) 
        #     [22, -1, 0,0] # tail end (not use)
        # ]
    #     keynum = len(self.optimize_pair)
    #     keypoints = np.zeros((keynum, 3), dtype=np.float32)
    #     for i in range(keynum):
    #         if self.optimize_pair[i][1] == 0:
    #             keypoints[i] = self.posed_joints[self.optimize_pair[i][2]]
    #         elif self.optimize_pair[i][1] == 1:
    #             keypoints[i] = self.posed_vertices[self.optimize_pair[i][2]]
    #     return keypoints


if __name__ == "__main__":
    bm = BodyModelNumpy()
    # filename = "H:/pig_results_anchor_sil/state_smth/pig_{}_frame_{:06d}.txt".format(0, 750)
    # filename = "D:/results/paper_teaser/demo6/state/pig_2_frame_077888.txt"
    # trans, poseparam, scale = bm.readstate(filename) 
    # bm.forward(poseparam, trans=trans, scale = scale) 
    # bm.write_obj("tmp/mesh2.obj")

    bm.write_obj("v_mouse.obj")