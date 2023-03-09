import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
from .networks import RGBDepthNet, weights_init, \
    SimpleAudioDepthNet, attentionNet, MaterialPropertyNet, RGBSpecNet, attentionSpecNet, RGBMaterialSpecNet,attentionRGBSpecNet, attentionDepthSpecNet, attentionDecoderNet, attentionTensformerEncoderSpecNet

class ModelBuilder():
    # builder for audio stream
    def build_audiodepth(self, audio_shape=[2,257,166], weights=''):
        net = SimpleAudioDepthNet(8, audio_shape=audio_shape, audio_feature_length=512)
        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for audio stream')
            net.load_state_dict(torch.load(weights))
        return net

    #builder for visual stream
    def build_rgbdepth(self, ngf=64, input_nc=3, output_nc=1, weights=''):
        
        net = RGBDepthNet(ngf, input_nc, output_nc)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net
    
    def build_rgbspec(self, ngf=64, input_nc=3, output_nc=2, weights=''):
        
        net = RGBSpecNet(ngf, input_nc, output_nc)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net
    
    def build_rgbmaterialspec(self, ngf=64, input_nc=3, output_nc=2, weights=''):
        
        net = RGBMaterialSpecNet(ngf, input_nc, output_nc)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net
    
    def build_attentionRGBSpecNet(self, weights=''):
        
        net = attentionRGBSpecNet(att_out_nc=512, input_nc=512)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net
    
    def build_attentionDepthSpecNet(self, weights=''):
        
        net = attentionDepthSpecNet(att_out_nc=512, input_nc=512)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net
    
        
    def build_depthspec(self, ngf=64, input_nc=1, output_nc=2, weights=''):
        
        net = RGBSpecNet(ngf, input_nc, output_nc)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net

    def build_attention(self, weights=''):
        
        net = attentionNet(att_out_nc=512, input_nc=2*512)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for attention stream')
            net.load_state_dict(torch.load(weights))
        return net
    
    def build_attention_spec(self, weights=''):
        
        net = attentionSpecNet(att_out_nc=512, input_nc=2*512)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for attention stream')
            net.load_state_dict(torch.load(weights))
        return net

    def build_attention_transformer_encoder_spec(self, weights=''):
        
        net = attentionTensformerEncoderSpecNet(att_out_nc=512, input_nc=3*512)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for attention stream')
            net.load_state_dict(torch.load(weights))
        return net
    
    def build_attention_decoder(self, weights=''):
        
        net = attentionDecoderNet(att_out_nc=512, input_nc=2*512)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for attention stream')
            net.load_state_dict(torch.load(weights))
        return net

    def build_material_property(self, nclass=10, weights='', init_weights=''):
        if len(init_weights) > 0:
            original_resnet = torchvision.models.resnet18(pretrained=True)
            net = MaterialPropertyNet(23, original_resnet)
            pre_trained_dict = torch.load(init_weights)['state_dict']
            pre_trained_mod_dict = OrderedDict()
            for k,v in pre_trained_dict.items():
                new_key = '.'.join(k.split('.')[1:])
                pre_trained_mod_dict[new_key] = v
            pre_trained_mod_dict = {k: v for k, v in pre_trained_mod_dict.items() if k in net.state_dict()}
            net.load_state_dict(pre_trained_mod_dict, strict=False)
            
            print('Initial Material Property Net Loaded')
            net.fc = nn.Linear(512, nclass)
        else:
            original_resnet = torchvision.models.resnet18(pretrained=False)
            net = MaterialPropertyNet(nclass, original_resnet)
            net.apply(weights_init)
            print('Moaterial Propert Net loaded')
        
        if len(weights) > 0:
            print('Loading weights for material property stream')
            net.load_state_dict(torch.load(weights))
        return net

if __name__ == "__main__":
    builder = ModelBuilder()
    net_audiodepth = builder.build_audiodepth()
    net_rgbspec = builder.build_rgbspec()
    net_attention = builder.build_attention_spec()
    net_depthspec = builder.build_depthspec()
    net_material = builder.build_material_property(weights="/home/xiaohu/workspace/my_habitat_transformer_image2reverb/ckpt-saved/material_replica.pth")
    
    input = {}
    input['img'] = torch.rand((4,3,128,128))
    input['audio'] = torch.rand((4,2,257,166))
    input['depth'] = torch.rand((4,1,128,128))
    
    depth_prediction1, rgbdepth_conv5feature = net_rgbspec(input['img']) # 4,1,128,128  # 4,512,4,4
    x, feat = net_material(input['img'])
    depth_prediction2, depth_conv5feature = net_depthspec(input['depth'])

    #audio_feat = audio_feat.repeat(1, 1, img_feat.shape[-2], img_feat.shape[-1]) #tile audio feature
    alpha, _ = net_attention(rgbdepth_conv5feature, depth_conv5feature, feat)
    depth_prediction = ((alpha*depth_prediction1)+((1-alpha)*depth_prediction2)) 