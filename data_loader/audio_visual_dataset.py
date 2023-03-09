import os.path
import time
import librosa
import h5py
import random
import math
import numpy as np
import glob
import torch
import pickle
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch.utils.data as data
import librosa
import matplotlib.pyplot as plt
import librosa.display


def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples

HOP_LENGTH = 62 
N_FFT = 512
WIN_LENGTH = 248 
"""
HOP_LENGTH = 100 #62 
N_FFT = 400 #511 
WIN_LENGTH = 200 #248 

"""
def generate_spectrogram(audioL, audioR, winl=WIN_LENGTH):
    #channel_1_spec = librosa.stft(audioL)#, n_fft=NFFT, win_length=winl)
    #channel_2_spec = librosa.stft(audioR)#, n_fft=NFFT, win_length=winl)
    #channel_1_spec = librosa.feature.melspectrogram(audioL, n_fft=512//2, win_length=winl)
    #channel_2_spec = librosa.feature.melspectrogram(audioR, n_fft=512//2, win_length=winl)
    channel_1_spec = librosa.stft(audioL, n_fft=N_FFT, win_length=winl, hop_length = HOP_LENGTH)
    channel_2_spec = librosa.stft(audioR, n_fft=N_FFT, win_length=winl, hop_length = HOP_LENGTH)
    spectro_two_channel = np.concatenate((np.expand_dims(np.abs(channel_1_spec), axis=0), np.expand_dims(np.abs(channel_2_spec), axis=0)), axis=0)
    #spectro_two_channel = np.log(spectro_two_channel + 1e-8)
    #spectro_two_channel = (spectro_two_channel - spectro_two_channel.min())/(spectro_two_channel.max() - spectro_two_channel.min())
    #print(spectro_two_channel.shape)

    """
    fig, ax = plt.subplots(1,2)                                               
    ax[0].set_title('Power spectrogram gt debug left')
    img_left = librosa.display.specshow(librosa.amplitude_to_db(spectro_two_channel[0,:,:]), 
                                y_axis='log', x_axis='time', ax=ax[0], ) #sr = 44100, n_fft = NFFT, win_length = winl)
    fig.colorbar(img_left, ax=ax[0], format="%+2.0f dB")
    # plt.savefig('/home/xiaohu/workspace/my_habitat/img/pred_IR_train{}.png')

    ax[0].set_title('Power spectrogram gt debug right')
    img_right = librosa.display.specshow(librosa.amplitude_to_db(spectro_two_channel[1,:,:]), 
                                y_axis='log', x_axis='time', ax=ax[1], )#sr = 44100, n_fft = NFFT, win_length = winl)
    fig.colorbar(img_right, ax=ax[1], format="%+2.0f dB")
    
    plt.savefig( '/home/xiaohu/workspace/beyond-depth-to-echos/data_loader/spec_debugs/IR_debug.jpg')
    
    print("shape:",spectro_two_channel.shape)
    """
    return spectro_two_channel #shape: (2, 256, 43)

def process_image(rgb, augment):
    if augment:
        # print('Doing Augmentation')
        enhancer = ImageEnhance.Brightness(rgb)
        rgb = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(rgb)
        rgb = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Contrast(rgb)
        rgb = enhancer.enhance(random.random()*0.6 + 0.7)
    return rgb


def parse_all_data(root_path, scenes):
    data_idx_all = []
    print(root_path)
    with open(root_path, 'rb') as f:
        data_dict = pickle.load(f)
    for scene in scenes:
        print(scene)
        data_idx_all += ['/'.join([scene, str(loc), str(ori)]) \
            for (loc,ori) in list(data_dict[scene].keys())]
        print(len(data_idx_all))    
    
    return data_idx_all, data_dict

def parse_all_data_my(root_path, scenes): #root path: /sas1/Sascha/SharedDatasets/sound-spaces/data/scene_observations/replica
    data_idx_all = []
    data_dict = {}
    print(root_path)
    #with open(root_path, 'rb') as f:
    #    data_dict = pickle.load(f)
    for scene in scenes:
        with open(os.path.join[root_path, scene, '.pkl'], 'rb') as f:
            data_dict_tmp = pickle.load(f)
            data_dict.update(data_dict_tmp)
        print(scene)
        data_idx_all += ['/'.join([scene, str(loc), str(ori)]) \
            for (loc,ori) in list(data_dict_tmp.keys())]
        print(len(data_idx_all))    
    
    return data_idx_all, data_dict
        
class AudioVisualDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        if self.opt.dataset == 'mp3d':
            self.data_idx, self.data = parse_all_data(
                                        os.path.join(self.opt.img_path, opt.mode+'.pkl'),
                                        self.opt.scenes[opt.mode])
            self.win_length = 32
        if self.opt.dataset == 'replica':
            self.data_idx, self.data = parse_all_data(self.opt.img_path, 
                                                        self.opt.scenes[opt.mode])   
            self.win_length = 128 #64
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.base_audio_path = self.opt.audio_path
        if self.opt.dataset == 'mp3d':
            self.audio_type = '3ms_sweep_16khz'
        if self.opt.dataset == 'replica':
            self.audio_type = '3ms_sweep'

    def __getitem__(self, index):
        #load audio
        scene, loc, orn = self.data_idx[index].split('/')
        audio_path = os.path.join(self.base_audio_path, scene, self.audio_type, orn, loc+'.wav')
        audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate, mono=False, duration=self.opt.audio_length)
        if self.opt.audio_normalize:
            audio = normalize(audio)

        #get the spectrogram of both channel
        audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0,:], audio[1,:]))
        #get the rgb image and depth image
        img = Image.fromarray(self.data[scene][(int(loc),int(orn))][('rgb')]).convert('RGB')
        
        if self.opt.mode == "train":
           img = process_image(img, self.opt.enable_img_augmentation)

        if self.opt.image_transform:
            img = self.vision_transform(img)

        depth = torch.FloatTensor(self.data[scene][(int(loc),int(orn))][('depth')])
        depth = depth.unsqueeze(0)
        
        if self.opt.mode == "train":
            if self.opt.enable_cropping:
                RESOLUTION = self.opt.image_resolution
                w_offset =  RESOLUTION - 128
                h_offset = RESOLUTION - 128
                left = random.randrange(0, w_offset + 1)
                upper = random.randrange(0, h_offset + 1)
                img = img[:, left:left+128,upper:upper+128]
                depth = depth[:, left:left+128,upper:upper+128]
        
        
        return {'img': img, 'depth':depth, 'audio':audio_spec_both}

    def __len__(self):
        return len(self.data_idx)

    def name(self):
        return 'AudioVisualDataset'
    
    
if __name__ == "__main__":
    root_path = "/home/xiaohu/workspace/data/scene_observations_128.pkl"