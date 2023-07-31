from PIL import Image
import torch
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('./') # add relative path
import open3d as o3d
from plyfile import PlyData, PlyElement

from module.sttr import STTR
from dataset.preprocess import normalization, compute_left_occ_region
from utilities.misc import NestedTensor

# save ply
def write_ply(points, filename):
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)

if __name__ == '__main__':
    # Default parameters
    args = type('', (), {})() # create empty args
    args.channel_dim = 128
    args.position_encoding='sine1d_rel'
    args.num_attn_layers=6
    args.nheads=8
    args.regression_head='ot'
    args.context_adjustment_layer='cal'
    args.cal_num_blocks=8
    args.cal_feat_dim=16
    args.cal_expansion_ratio=4

    model = STTR(args).cuda().eval()

    # Load the pretrained model
    model_file_name = "kitti_finetuned_model.pth.tar"
    checkpoint = torch.load(model_file_name)
    pretrained_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_dict, strict=False) # prevent BN parameters from breaking the model loading
    print("Pre-trained model successfully loaded.")

    # ['006975', '007770', ...] 从低到高排序
    scan_names = sorted(list(set([os.path.basename(x)[0:4] \
        for x in os.listdir('sample_data/dataset_1/left/images/')])))

    for i in range(len(scan_names)):
        scan_name = scan_names[i]
        left = np.array(Image.open('sample_data/dataset_1/left/images/' + scan_name + '.png'))
        right = np.array(Image.open('sample_data/dataset_1/right/images/' + scan_name + '.png'))
        mask = np.array(Image.open('sample_data/dataset_1/left/masks_gt/' + scan_name + '.png'))  # 0 or 255
        # normalize
        input_data = {'left': left, 'right':right}
        input_data = normalization(**input_data)

        # donwsample attention by stride of 3
        h, w, _ = left.shape
        bs = 1
        downsample = 1
        col_offset = int(downsample / 2)  # 1
        row_offset = int(downsample / 2)
        sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).cuda()
        sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).cuda()
        
        # build NestedTensor
        input_data = NestedTensor(input_data['left'].cuda()[None,],input_data['right'].cuda()[None,],
            sampled_cols=sampled_cols, sampled_rows=sampled_rows)
        output = model(input_data)

        # set disparity of occ area to 0
        disp_pred = output['disp_pred'].data.cpu().numpy()[0]
        occ_pred = output['occ_pred'].data.cpu().numpy()[0] > 0.5
        disp_pred[occ_pred] = 0.0
        plt.imsave('output/' + scan_name + '.png', disp_pred)
        # target_region = mask < 125
        # disp_pred[target_region] = 0.0

        # # camera intrinsic
        # A_cam = np.array([[1194.072289*0.25, 0.0, 960*0.25],
        #                 [0.0, 1194.072289*0.25, 540*0.25],
        #                 [0.0,     0.0, 1.0]], dtype=np.float32)
        # b_dis = 4.05

        # # calculate XYZ
        # points_Z0 = A_cam[0,0]*b_dis/(disp_pred)
        # u_mesh = np.linspace(1,disp_pred.shape[1],disp_pred.shape[1])[None,:]
        # u_mesh = np.concatenate([u_mesh for i in range(disp_pred.shape[0])],axis=0)
        # points_X0 = points_Z0*(u_mesh-A_cam[0,2])/A_cam[0,0]
        # v_mesh = np.linspace(1,disp_pred.shape[0],disp_pred.shape[0])[:,None]
        # v_mesh = np.concatenate([v_mesh for i in range(disp_pred.shape[1])],axis=1)
        # points_Y0 = points_Z0*(v_mesh-A_cam[1,2])/A_cam[1,1]
        # points = np.concatenate((points_X0.flatten()[:,None],points_Y0.flatten()[:,None],points_Z0.flatten()[:,None]), axis=1)

        # points_mask = (points[:,2]>5) & (points[:,2]<150)
        # points = points[points_mask,:]

        # write_ply(points, 'output/' + scan_name + '.ply')
        print('finish: ' + scan_name)