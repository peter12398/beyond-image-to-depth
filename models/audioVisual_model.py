import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from . import networks,criterion
from torch.autograd import Variable

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_rgbspec, self.net_depthspec, self.net_attention, self.net_material = nets
        

    def forward(self, input, volatile=False):
        rgb_input = input['img']
        #audio_input = input['audio']/input['audio'].max()
        audio_input = input['audio']
        depth_gt = input['depth']

        #print("audio_input shape",audio_input.shape)
        depth_spec, depth_feat = self.net_depthspec(depth_gt) # torch.Size([256, 2, 257, 166])  torch.Size([256, 512, 1, 1])
        img_spec, img_feat = self.net_rgbspec(rgb_input) # torch.Size([256, 1, 128, 128]) torch.Size([256, 512, 4, 4])
        
        material_class, material_feat = self.net_material(rgb_input) 
        material_feat = material_feat.unsqueeze(-1).unsqueeze(-1)
        #alpha_rgb, _ = self.attentionRGBSpecNet(img_feat,material_feat)
        #alpha_depth, _ = self.attentionDepthSpecNet(depth_feat,material_feat)
        #img_spec, img_feat = self.net_rgbmaterial(rgb_input,material_feat)
        #audio_feat = audio_feat.repeat(1, 1, img_feat.shape[-2], img_feat.shape[-1]) #tile audio feature
        
        #alpha, _ = self.net_attention(img_feat, depth_feat, material_feat)
        alpha, _ = self.net_attention(img_feat, depth_feat, material_feat)
        #spec_prediction = depth_spec #((alpha*img_spec)+((1-alpha)*depth_spec)) 
        spec_prediction = alpha
        #spec_prediction = ((alpha_rgb*img_spec)+((alpha_depth)*depth_spec)) 
        #spec_prediction = depth_spec
        #print("spec_prediction max:",spec_prediction.max())
        #print("spec_prediction min:",spec_prediction.min())
        #print("audio_input max:",audio_input.max())
        #print("audio_input min:",audio_input.min())

        
        output =  {'img_spec': img_spec ,
                    'depth_spec': depth_spec ,
                    'spec_predicted': spec_prediction , 
                    'attention': alpha,
                    'img': rgb_input,
                    'spec_gt': audio_input,
                    'depth': depth_gt}
        return output

