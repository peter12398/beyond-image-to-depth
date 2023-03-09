import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])

def unet_upconv2(input_nc, output_nc, kernel_size=4, stride=2, padding=1, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])

        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class SimpleAudioDepthNet(nn.Module):
    ## strucure adapted from VisualEchoes [ECCV 2020]
    r"""A Simple 3-Conv CNN followed by a fully connected layer
    """
    def __init__(self, conv1x1_dim, audio_shape, audio_feature_length, output_nc=1):
        super(SimpleAudioDepthNet, self).__init__()
        self._n_input_audio = audio_shape[0]
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.conv1 = create_conv(self._n_input_audio, 32, kernel=self._cnn_layers_kernel_size[0], paddings=0, stride=self._cnn_layers_stride[0])
        self.conv2 = create_conv(32, 64, kernel=self._cnn_layers_kernel_size[1], paddings=0, stride=self._cnn_layers_stride[1])
        self.conv3 = create_conv(64, conv1x1_dim, kernel=self._cnn_layers_kernel_size[2], paddings=0, stride=self._cnn_layers_stride[2])
        layers = [self.conv1, self.conv2, self.conv3]
        self.feature_extraction = nn.Sequential(*layers)
        self.conv1x1 = create_conv(conv1x1_dim * cnn_dims[0] * cnn_dims[1], audio_feature_length, 1, 0)

        self.rgbdepth_upconvlayer1 = unet_upconv(512, 512) #1016 (audio-visual feature) = 512 (visual feature) + 504 (audio feature)
        self.rgbdepth_upconvlayer2 = unet_upconv(512, 256)
        self.rgbdepth_upconvlayer3 = unet_upconv(256, 128)
        self.rgbdepth_upconvlayer4 = unet_upconv(128, 64)
        self.rgbdepth_upconvlayer5 = unet_upconv(64, 32)
        self.rgbdepth_upconvlayer6 = unet_upconv(32, 16)
        self.rgbdepth_upconvlayer7 = unet_upconv(16, output_nc, True)

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.conv1x1(x)
        
        audio_feat = x
        
        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(audio_feat)
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(rgbdepth_upconv1feature)
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(rgbdepth_upconv2feature)
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(rgbdepth_upconv3feature)
        rgbdepth_upconv5feature = self.rgbdepth_upconvlayer5(rgbdepth_upconv4feature)
        rgbdepth_upconv6feature = self.rgbdepth_upconvlayer6(rgbdepth_upconv5feature)
        depth_prediction = self.rgbdepth_upconvlayer7(rgbdepth_upconv6feature)
        return depth_prediction, audio_feat

class attentionNet(nn.Module):
    def __init__(self, att_out_nc, input_nc):
        super(attentionNet, self).__init__()
        #initialize layers
        
        self.attention_img = nn.Bilinear(512, 512, att_out_nc)
        self.attention_material = nn.Bilinear(512, 512, att_out_nc)
        self.upconvlayer1 = unet_upconv(input_nc, 512) 
        self.upconvlayer2 = unet_upconv(512, 256)
        self.upconvlayer3 = unet_upconv(256, 128)
        self.upconvlayer4 = unet_upconv(128, 64)
        self.upconvlayer5 = unet_upconv(64, 1, True)
        
    def forward(self, rgb_feat, echo_feat, mat_feat):
        rgb_feat = rgb_feat.permute(0, 2, 3, 1).contiguous()
        echo_feat = echo_feat.permute(0, 2, 3, 1).contiguous()
        mat_feat = mat_feat.permute(0, 2, 3, 1).contiguous()
        
        attentionImg = self.attention_img(rgb_feat, echo_feat)
        attentionMat = self.attention_material(mat_feat, echo_feat)
    
        attentionImg = attentionImg.permute(0, 3, 1, 2).contiguous()
        attentionMat = attentionMat.permute(0, 3, 1, 2).contiguous()
        
        audioVisual_feature = torch.cat((attentionImg, attentionMat), dim=1)
        
        upconv1feature = self.upconvlayer1(audioVisual_feature)
        upconv2feature = self.upconvlayer2(upconv1feature)
        upconv3feature = self.upconvlayer3(upconv2feature)
        upconv4feature = self.upconvlayer4(upconv3feature)
        attention = self.upconvlayer5(upconv4feature)
        return attention, audioVisual_feature

class attentionDecoderNet(nn.Module):
    def __init__(self, att_out_nc, input_nc):
        super(attentionDecoderNet, self).__init__()
        #initialize layers
        

        self.upconvlayer1 = unet_upconv(input_nc, 512) 
        self.upconvlayer2 = unet_upconv(512, 256)
        self.upconvlayer3 = unet_upconv(256, 128)
        self.upconvlayer4 = unet_upconv(128, 64)
        self.upconvlayer5 = unet_upconv(64, 32)
        self.upconvlayer6 = unet_upconv(32, 16)
        self.upconvlayer7 = unet_upconv(16, 8)
        self.upconvlayer8 = unet_upconv2(8, 4, kernel_size=[5,41], stride=[2,1], padding=1)
        self.upconvlayer9 = unet_upconv2(4, 2, kernel_size=3, stride=1, padding=1, outermost=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
    def forward(self, rgb_feat, echo_feat, mat_feat): #audio_shape=[2,257,166]
        #rgb_feat = rgb_feat.permute(0, 2, 3, 1).contiguous()
        #echo_feat = echo_feat.permute(0, 2, 3, 1).contiguous()
        #mat_feat = mat_feat.permute(0, 2, 3, 1).contiguous()
        
        #material_feat = mat_feat.reshape((mat_feat.shape[0],-1)).unsqueeze(0)
        #audio_feat = echo_feat.reshape((echo_feat.shape[0],-1)).unsqueeze(0)
        #img_feat = rgb_feat.reshape((rgb_feat.shape[0],-1)).unsqueeze(0)
        
        rgb_feat = rgb_feat.squeeze(-1).squeeze(-1).unsqueeze(0)
        echo_feat = echo_feat.squeeze(-1).squeeze(-1).unsqueeze(0)
        mat_feat = mat_feat.unsqueeze(0)
        
        src = torch.cat([rgb_feat,echo_feat,mat_feat],dim=0)
        out = self.transformer_encoder(src)
        #out = torch.cat([out[0:16],out[16:32],out[32:]],dim=-1)
        
        #attentionImg = self.attention_img(rgb_feat, echo_feat)
        #attentionMat = self.attention_material(mat_feat, echo_feat)
    
        #attentionImg = attentionImg.permute(0, 3, 1, 2).contiguous()
        #attentionMat = attentionMat.permute(0, 3, 1, 2).contiguous()
        
        #audioVisual_feature = out.unsqueeze(-1).unsqueeze(-1) #torch.cat((attentionImg, attentionMat), dim=1)
        #audioVisual_feature = out.reshape((out.shape[0],out.shape[1],mat_feat.shape[1],mat_feat.shape[2],mat_feat.shape[3]))
        audioVisual_feature = out
        audioVisual_feature = torch.cat([audioVisual_feature[0],audioVisual_feature[1]],dim=1).unsqueeze(-1).unsqueeze(-1)
        
        upconv1feature = self.upconvlayer1(audioVisual_feature)
        upconv2feature = self.upconvlayer2(upconv1feature)
        upconv3feature = self.upconvlayer3(upconv2feature)
        upconv4feature = self.upconvlayer4(upconv3feature)
        upconv5feature = self.upconvlayer5(upconv4feature)
        upconv6feature = self.upconvlayer6(upconv5feature)
        upconv7feature = self.upconvlayer7(upconv6feature)
        upconv8feature = self.upconvlayer8(upconv7feature)
        attention = self.upconvlayer9(upconv8feature)
        #attention = self.upconvlayer7(upconv9feature)
        return attention, audioVisual_feature


class attentionSpecNet(nn.Module):
    def __init__(self, att_out_nc, input_nc):
        super(attentionSpecNet, self).__init__()
        #initialize layers
        
        self.attention_img = nn.Bilinear(512, 512, att_out_nc)
        self.attention_material = nn.Bilinear(512, 512, att_out_nc)
        self.upconvlayer1 = unet_upconv(input_nc, 512) 
        self.upconvlayer2 = unet_upconv(512, 256)
        self.upconvlayer3 = unet_upconv(256, 128)

        """
        self.rgbdepth_upconvlayer4 = unet_upconv2(ngf * 4, ngf, kernel_size=[4,8], stride=[2,1], padding=1) #unet_upconv(ngf * 4, ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv2(ngf, ngf//2, kernel_size=[4,8], stride=[1,1], padding=1)  #unet_upconv(ngf * 2, ngf)
        self.rgbdepth_upconvlayer6 = unet_upconv2(ngf//2, ngf//4, kernel_size=3, stride=1, padding=1)
        self.rgbdepth_upconvlayer7 = unet_upconv2(ngf//4, output_nc, kernel_size=3, stride=1, padding=1, outermost=True)
        """
        self.upconvlayer4 = unet_upconv(128, 64) #unet_upconv(128, 64)
        self.upconvlayer5 = unet_upconv2(64, 32, kernel_size=[4,6], stride=[2,2], padding=1) #unet_upconv(64, 32)
        """
        self.rgbdepth_upconvlayer4 = unet_upconv2(ngf * 4, ngf, kernel_size=[4,6], stride=[2,1], padding=1) #unet_upconv(ngf * 4, ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv2(ngf, ngf//2, kernel_size=[4,6], stride=[2,1], padding=1)  #unet_upconv(ngf * 2, ngf)
        self.rgbdepth_upconvlayer6 = unet_upconv2(ngf//2, ngf//4, kernel_size=[4,8], stride=[2,1], padding=1)
        """
        self.upconvlayer6 = unet_upconv2(32, 16, kernel_size=[4,6], stride=[2,1], padding=1)
        self.upconvlayer7 = unet_upconv2(16, 8, kernel_size=[4,6], stride=[2,1], padding=1)
        self.upconvlayer8 = unet_upconv2(8, 1, kernel_size=[4,6], stride=[2,1], padding=1,outermost=True)
        #self.upconvlayer5 = unet_upconv(64, 1, True)
        
    def forward(self, rgb_feat, echo_feat, mat_feat):
        rgb_feat = rgb_feat.permute(0, 2, 3, 1).contiguous()
        echo_feat = echo_feat.permute(0, 2, 3, 1).contiguous()
        mat_feat = mat_feat.permute(0, 2, 3, 1).contiguous()
        
        attentionImg = self.attention_img(rgb_feat, echo_feat)
        attentionMat = self.attention_material(mat_feat, echo_feat)
        #attentionImg = self.attention_img(rgb_feat, mat_feat)
        #attentionMat = self.attention_material(mat_feat, echo_feat)
    
        attentionImg = attentionImg.permute(0, 3, 1, 2).contiguous()
        attentionMat = attentionMat.permute(0, 3, 1, 2).contiguous()
        
        audioVisual_feature = torch.cat((attentionImg, attentionMat), dim=1)
        #shape: (2, 256, 43)
        upconv1feature = self.upconvlayer1(audioVisual_feature)
        upconv2feature = self.upconvlayer2(upconv1feature)
        upconv3feature = self.upconvlayer3(upconv2feature)
        upconv4feature = self.upconvlayer4(upconv3feature)
        upconv5feature = self.upconvlayer5(upconv4feature)
        upconv6feature = self.upconvlayer6(upconv5feature)
        upconv7feature = self.upconvlayer7(upconv6feature)
        attention = self.upconvlayer8(upconv7feature)
        return attention, audioVisual_feature


class attentionTensformerEncoderSpecNet(nn.Module):
    def __init__(self, att_out_nc, input_nc):
        super(attentionTensformerEncoderSpecNet, self).__init__()
        #initialize layers
        
        self.attention_img = nn.Bilinear(512, 512, att_out_nc)
        self.attention_material = nn.Bilinear(512, 512, att_out_nc)
        self.upconvlayer1 = unet_upconv(input_nc, 512) 
        self.upconvlayer2 = unet_upconv(512, 256)
        self.upconvlayer3 = unet_upconv(256, 128)

        """
        self.rgbdepth_upconvlayer4 = unet_upconv2(ngf * 4, ngf, kernel_size=[4,8], stride=[2,1], padding=1) #unet_upconv(ngf * 4, ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv2(ngf, ngf//2, kernel_size=[4,8], stride=[1,1], padding=1)  #unet_upconv(ngf * 2, ngf)
        self.rgbdepth_upconvlayer6 = unet_upconv2(ngf//2, ngf//4, kernel_size=3, stride=1, padding=1)
        self.rgbdepth_upconvlayer7 = unet_upconv2(ngf//4, output_nc, kernel_size=3, stride=1, padding=1, outermost=True)
        """
        self.upconvlayer4 = unet_upconv(128, 64) #unet_upconv(128, 64)
        self.upconvlayer5 = unet_upconv2(64, 32, kernel_size=[4,4], stride=[2,2], padding=1) #unet_upconv(64, 32)
        """
        self.rgbdepth_upconvlayer4 = unet_upconv2(ngf * 4, ngf, kernel_size=[4,6], stride=[2,1], padding=1) #unet_upconv(ngf * 4, ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv2(ngf, ngf//2, kernel_size=[4,6], stride=[2,1], padding=1)  #unet_upconv(ngf * 2, ngf)
        self.rgbdepth_upconvlayer6 = unet_upconv2(ngf//2, ngf//4, kernel_size=[4,8], stride=[2,1], padding=1)
        """
        self.upconvlayer6 = unet_upconv2(32, 32, kernel_size=[4,4], stride=[2,2], padding=1)
        self.upconvlayer7 = unet_upconv2(32, 16, kernel_size=[4,4], stride=[2,1], padding=1)
        self.upconvlayer8 = unet_upconv2(16, 16, kernel_size=[4,4], stride=[2,1], padding=1)
        self.upconvlayer9 = unet_upconv2(16, 2, kernel_size=[4,4], stride=[1,1], padding=1,outermost=True)
        #self.upconvlayer5 = unet_upconv(64, 1, True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
      
        
    def forward(self, rgb_feat, echo_feat, mat_feat):
        rgb_feat = rgb_feat.permute(0, 2, 3, 1).contiguous()
        echo_feat = echo_feat.permute(0, 2, 3, 1).contiguous()
        mat_feat = mat_feat.permute(0, 2, 3, 1).contiguous()

        
        rgb_feat = rgb_feat.squeeze(-2).squeeze(-2).unsqueeze(0)
        echo_feat = echo_feat.squeeze(-2).squeeze(-2).unsqueeze(0)
        mat_feat = mat_feat.squeeze(-2).squeeze(-2).unsqueeze(0)
        """
        
        ###########
        attentionImg = self.attention_img(rgb_feat, echo_feat)
        attentionMat = self.attention_material(mat_feat, echo_feat)
        #attentionImg = self.attention_img(rgb_feat, mat_feat)
        #attentionMat = self.attention_material(mat_feat, echo_feat)
    
        attentionImg = attentionImg.permute(0, 3, 1, 2).contiguous()
        attentionMat = attentionMat.permute(0, 3, 1, 2).contiguous()
        
        audioVisual_feature = torch.cat((attentionImg, attentionMat), dim=1)
        #############
        """

        
        src = torch.cat([rgb_feat,echo_feat,mat_feat],dim=0)
        #out = self.transformer_encoder(src)
        #audioVisual_feature = out
        audioVisual_feature = src

        """
        attentionImg = self.attention_img(rgb_feat, echo_feat)
        attentionMat = self.attention_material(mat_feat, echo_feat)
        #attentionImg = self.attention_img(rgb_feat, mat_feat)
        #attentionMat = self.attention_material(mat_feat, echo_feat)
    
        attentionImg = attentionImg.permute(0, 3, 1, 2).contiguous()
        attentionMat = attentionMat.permute(0, 3, 1, 2).contiguous()
        
        audioVisual_feature = torch.cat((attentionImg, attentionMat), dim=1)
        
        
        #audioVisual_feature = torch.cat([audioVisual_feature[0],audioVisual_feature[1],audioVisual_feature[2]],dim=1).unsqueeze(-1).unsqueeze(-1) #audioVisual_feature[1].unsqueeze(-1).unsqueeze(-1) #torch.cat([audioVisual_feature[0],audioVisual_feature[1]],dim=1).unsqueeze(-1).unsqueeze(-1)
        audioVisual_feature = torch.cat([audioVisual_feature[2]],dim=1).unsqueeze(-1).unsqueeze(-1)
        """

        audioVisual_feature = torch.cat([audioVisual_feature[0],audioVisual_feature[1],audioVisual_feature[2]],dim=1).unsqueeze(-1).unsqueeze(-1)

        #shape: (2, 256, 43) -> shape: (2, 256, 162) -> shape: (2, 257, 72)
        upconv1feature = self.upconvlayer1(audioVisual_feature) # -> torch.Size([128, 512, 2, 2])
        upconv2feature = self.upconvlayer2(upconv1feature) # -> torch.Size([128, 256, 4, 4])
        upconv3feature = self.upconvlayer3(upconv2feature) # -> torch.Size([128, 128, 8, 8])
        upconv4feature = self.upconvlayer4(upconv3feature) # -> torch.Size([128, 64, 16, 16])
        upconv5feature = self.upconvlayer5(upconv4feature) #-> torch.Size([128, 32, 32, 34])
        upconv6feature = self.upconvlayer6(upconv5feature)
        upconv7feature = self.upconvlayer7(upconv6feature)
        upconv8feature = self.upconvlayer8(upconv7feature)
        attention = self.upconvlayer9(upconv8feature)
        return attention, audioVisual_feature


class attentionRGBSpecNet(nn.Module):
    def __init__(self, att_out_nc, input_nc):
        super(attentionRGBSpecNet, self).__init__()
        #initialize layers
        
        self.attention_img = nn.Bilinear(512, 512, att_out_nc)
        #self.attention_material = nn.Bilinear(512, 512, att_out_nc)
        self.upconvlayer1 = unet_upconv(input_nc, 512) 
        self.upconvlayer2 = unet_upconv(512, 256)
        self.upconvlayer3 = unet_upconv(256, 128)
        self.upconvlayer4 = unet_upconv(128, 64)
        self.upconvlayer5 = unet_upconv(64, 32)
        self.upconvlayer6 = unet_upconv2(32, 16, kernel_size=[5,41], stride=[2,1], padding=1)
        self.upconvlayer7 = unet_upconv2(16, 1, kernel_size=3, stride=1, padding=1, outermost=True)
        #self.upconvlayer5 = unet_upconv(64, 1, True)
        
    def forward(self, rgb_feat, mat_feat):
        rgb_feat = rgb_feat.permute(0, 2, 3, 1).contiguous()
        #echo_feat = echo_feat.permute(0, 2, 3, 1).contiguous()
        mat_feat = mat_feat.permute(0, 2, 3, 1).contiguous()
        
        attentionImg = self.attention_img(rgb_feat, mat_feat)
        #attentionMat = self.attention_material(mat_feat, echo_feat)
        #attentionImg = self.attention_img(rgb_feat, mat_feat)
        #attentionMat = self.attention_material(mat_feat, echo_feat)
    
        attentionImg = attentionImg.permute(0, 3, 1, 2).contiguous()
        #attentionMat = attentionMat.permute(0, 3, 1, 2).contiguous()
        
        #audioVisual_feature = torch.cat((attentionImg, attentionMat), dim=1)
        audioVisual_feature = attentionImg
        
        upconv1feature = self.upconvlayer1(audioVisual_feature)
        upconv2feature = self.upconvlayer2(upconv1feature)
        upconv3feature = self.upconvlayer3(upconv2feature)
        upconv4feature = self.upconvlayer4(upconv3feature)
        upconv5feature = self.upconvlayer5(upconv4feature)
        upconv6feature = self.upconvlayer6(upconv5feature)
        attention = self.upconvlayer7(upconv6feature)
        return attention, audioVisual_feature


class attentionDepthSpecNet(nn.Module):
    def __init__(self, att_out_nc, input_nc):
        super(attentionDepthSpecNet, self).__init__()
        #initialize layers
        
        self.attention_img = nn.Bilinear(512, 512, att_out_nc)
        #self.attention_material = nn.Bilinear(512, 512, att_out_nc)
        self.upconvlayer1 = unet_upconv(input_nc, 512) 
        self.upconvlayer2 = unet_upconv(512, 256)
        self.upconvlayer3 = unet_upconv(256, 128)
        self.upconvlayer4 = unet_upconv(128, 64)
        self.upconvlayer5 = unet_upconv(64, 32)
        self.upconvlayer6 = unet_upconv2(32, 16, kernel_size=[5,41], stride=[2,1], padding=1)
        self.upconvlayer7 = unet_upconv2(16, 1, kernel_size=3, stride=1, padding=1, outermost=True)
        #self.upconvlayer5 = unet_upconv(64, 1, True)
        
    def forward(self, rgb_feat, mat_feat):
        rgb_feat = rgb_feat.permute(0, 2, 3, 1).contiguous()
        #echo_feat = echo_feat.permute(0, 2, 3, 1).contiguous()
        mat_feat = mat_feat.permute(0, 2, 3, 1).contiguous()
        
        attentionImg = self.attention_img(rgb_feat, mat_feat)
        #attentionMat = self.attention_material(mat_feat, echo_feat)
        #attentionImg = self.attention_img(rgb_feat, mat_feat)
        #attentionMat = self.attention_material(mat_feat, echo_feat)
    
        attentionImg = attentionImg.permute(0, 3, 1, 2).contiguous()
        #attentionMat = attentionMat.permute(0, 3, 1, 2).contiguous()
        
        #audioVisual_feature = torch.cat((attentionImg, attentionMat), dim=1)
        audioVisual_feature = attentionImg
        
        upconv1feature = self.upconvlayer1(audioVisual_feature)
        upconv2feature = self.upconvlayer2(upconv1feature)
        upconv3feature = self.upconvlayer3(upconv2feature)
        upconv4feature = self.upconvlayer4(upconv3feature)
        upconv5feature = self.upconvlayer5(upconv4feature)
        upconv6feature = self.upconvlayer6(upconv5feature)
        attention = self.upconvlayer7(upconv6feature)
        return attention, audioVisual_feature


class RGBDepthNet(nn.Module):
    def __init__(self, ngf=64, input_nc=3, output_nc=1):
        super(RGBDepthNet, self).__init__()
        #initialize layers
        self.rgbdepth_convlayer1 = unet_conv(input_nc, ngf)
        self.rgbdepth_convlayer2 = unet_conv(ngf, ngf * 2)
        self.rgbdepth_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.rgbdepth_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.rgbdepth_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.rgbdepth_upconvlayer1 = unet_upconv(512, ngf * 8)
        self.rgbdepth_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.rgbdepth_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.rgbdepth_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True)
        #self.conv1x1 = create_conv(512, 8, 1, 0) #reduce dimension of extracted visual features

    def forward(self, x):
        rgbdepth_conv1feature = self.rgbdepth_convlayer1(x)
        rgbdepth_conv2feature = self.rgbdepth_convlayer2(rgbdepth_conv1feature)
        rgbdepth_conv3feature = self.rgbdepth_convlayer3(rgbdepth_conv2feature)
        rgbdepth_conv4feature = self.rgbdepth_convlayer4(rgbdepth_conv3feature)
        rgbdepth_conv5feature = self.rgbdepth_convlayer5(rgbdepth_conv4feature)
        
        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(rgbdepth_conv5feature)
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(torch.cat((rgbdepth_upconv1feature, rgbdepth_conv4feature), dim=1))
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(torch.cat((rgbdepth_upconv2feature, rgbdepth_conv3feature), dim=1))
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(torch.cat((rgbdepth_upconv3feature, rgbdepth_conv2feature), dim=1))
        depth_prediction = self.rgbdepth_upconvlayer5(torch.cat((rgbdepth_upconv4feature, rgbdepth_conv1feature), dim=1))
        return depth_prediction, rgbdepth_conv5feature


class RGBSpecNet(nn.Module):
    def __init__(self, ngf=64, input_nc=3, output_nc=1):
        super(RGBSpecNet, self).__init__()
        #initialize layers
        self.rgbdepth_convlayer1 = unet_conv(input_nc, ngf)
        self.rgbdepth_convlayer2 = unet_conv(ngf, ngf * 2)
        self.rgbdepth_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.rgbdepth_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.rgbdepth_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.rgbdepth_upconvlayer1 = unet_upconv(512, ngf * 8)
        self.rgbdepth_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.rgbdepth_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.rgbdepth_upconvlayer4 = unet_upconv2(ngf * 4, ngf, kernel_size=[4,6], stride=[2,1], padding=1) #unet_upconv(ngf * 4, ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv2(ngf, ngf//2, kernel_size=[4,6], stride=[2,1], padding=1)  #unet_upconv(ngf * 2, ngf)
        self.rgbdepth_upconvlayer6 = unet_upconv2(ngf//2, ngf//4, kernel_size=[4,8], stride=[2,1], padding=1)
        self.rgbdepth_upconvlayer7 = unet_upconv2(ngf//4, output_nc, kernel_size=3, stride=1, padding=1, outermost=True)
        #self.conv1x1 = create_conv(512, 8, 1, 0) #reduce dimension of extracted visual features
        self.conv1x1 = unet_conv(ngf * 8, ngf * 8) #reduce dimension of extracted visual features
        self.conv1x1_2 = unet_conv(ngf * 8, ngf * 8)

    def forward(self, x):
        rgbdepth_conv1feature = self.rgbdepth_convlayer1(x) #  torch.Size([4, 64, 64, 64])
        rgbdepth_conv2feature = self.rgbdepth_convlayer2(rgbdepth_conv1feature) #torch.Size([4, 128, 32, 32])
        rgbdepth_conv3feature = self.rgbdepth_convlayer3(rgbdepth_conv2feature) #torch.Size([4, 256, 16, 16])
        rgbdepth_conv4feature = self.rgbdepth_convlayer4(rgbdepth_conv3feature) #torch.Size([4, 512, 8, 8])
        rgbdepth_conv5feature = self.rgbdepth_convlayer5(rgbdepth_conv4feature) #torch.Size([4, 512, 4, 4])
        rgbdepth_conv6feature = self.conv1x1(rgbdepth_conv5feature)
        rgbdepth_conv7feature = self.conv1x1_2(rgbdepth_conv6feature)
        #shape: (2, 256, 43)
        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(rgbdepth_conv5feature) #torch.Size([4, 512, 8, 8])
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(torch.cat((rgbdepth_upconv1feature, rgbdepth_conv4feature), dim=1))#4,1024,8,8 - torch.Size([4, 256, 16, 16])
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(torch.cat((rgbdepth_upconv2feature, rgbdepth_conv3feature), dim=1))#torch.Size([4, 128, 32, 32])
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(torch.cat((rgbdepth_upconv3feature, rgbdepth_conv2feature), dim=1))#torch.Size([4, 64, 64, 64])
        rgbdepth_upconv5feature = self.rgbdepth_upconvlayer5(rgbdepth_upconv4feature) #(torch.cat((rgbdepth_upconv4feature, rgbdepth_conv1feature), dim=1))#torch.Size([4, 2, 128, 128])
        rgbdepth_upconv6feature = self.rgbdepth_upconvlayer6(rgbdepth_upconv5feature)
        depth_prediction = self.rgbdepth_upconvlayer7(rgbdepth_upconv6feature)
        return depth_prediction, rgbdepth_conv7feature


class RGBMaterialSpecNet(nn.Module):
    def __init__(self, ngf=64, input_nc=3, output_nc=1):
        super(RGBMaterialSpecNet, self).__init__()
        #initialize layers
        self.rgbdepth_convlayer1 = unet_conv(input_nc, ngf)
        self.rgbdepth_convlayer2 = unet_conv(ngf, ngf * 2)
        self.rgbdepth_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.rgbdepth_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.rgbdepth_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.rgbdepth_upconvlayer1 = unet_upconv(512*2, ngf * 8)
        self.rgbdepth_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.rgbdepth_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.rgbdepth_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv(ngf * 2, ngf)
        self.rgbdepth_upconvlayer6 = unet_upconv2(ngf, ngf//2, kernel_size=[5,41], stride=[2,1], padding=1)
        self.rgbdepth_upconvlayer7 = unet_upconv2(ngf//2, output_nc, kernel_size=3, stride=1, padding=1, outermost=True)
        #self.conv1x1 = create_conv(512, 8, 1, 0) #reduce dimension of extracted visual features

    def forward(self, x, material_feature):
        rgbdepth_conv1feature = self.rgbdepth_convlayer1(x) #  torch.Size([4, 64, 64, 64])
        rgbdepth_conv2feature = self.rgbdepth_convlayer2(rgbdepth_conv1feature) #torch.Size([4, 128, 32, 32])
        rgbdepth_conv3feature = self.rgbdepth_convlayer3(rgbdepth_conv2feature) #torch.Size([4, 256, 16, 16])
        rgbdepth_conv4feature = self.rgbdepth_convlayer4(rgbdepth_conv3feature) #torch.Size([4, 512, 8, 8])
        rgbdepth_conv5feature = self.rgbdepth_convlayer5(rgbdepth_conv4feature) #torch.Size([4, 512, 4, 4])
        
        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(torch.cat([rgbdepth_conv5feature,material_feature],dim=1)) #torch.Size([4, 512, 8, 8])
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(torch.cat((rgbdepth_upconv1feature, rgbdepth_conv4feature), dim=1))#4,1024,8,8 - torch.Size([4, 256, 16, 16])
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(torch.cat((rgbdepth_upconv2feature, rgbdepth_conv3feature), dim=1))#torch.Size([4, 128, 32, 32])
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(torch.cat((rgbdepth_upconv3feature, rgbdepth_conv2feature), dim=1))#torch.Size([4, 64, 64, 64])
        rgbdepth_upconv5feature = self.rgbdepth_upconvlayer5(torch.cat((rgbdepth_upconv4feature, rgbdepth_conv1feature), dim=1))#torch.Size([4, 2, 128, 128])
        rgbdepth_upconv6feature = self.rgbdepth_upconvlayer6(rgbdepth_upconv5feature)
        depth_prediction = self.rgbdepth_upconvlayer7(rgbdepth_upconv6feature)
        return depth_prediction, rgbdepth_conv5feature


class MaterialPropertyNet(nn.Module):
    def __init__(self, nclass, backbone):
        super(MaterialPropertyNet, self).__init__()
        
        self.pretrained = backbone
        self.pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, nclass)

    def forward(self, x):
        # pre-trained ResNet feature
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        feat = self.pretrained.layer4(x)
        x = self.pool(feat)
        feat2 = x.view(-1, 512)
        x = self.fc(feat2)
        return x, feat2

if __name__ == "__main__":
    input_ = torch.rand((4,3,128,128)) #torch.Size([256, 2, 257, 166])
    model = RGBSpecNet(input_nc=3, output_nc=2)
    out = model(input_)
    print(out[0].shape)
    print(out[1].shape)
    