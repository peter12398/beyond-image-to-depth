#!/usr/bin/env python
import numpy as np
import os
import torch
import librosa
import matplotlib.pyplot as plt
import librosa.display

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    # select only the values that are greater than zero
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    if rmse != rmse:
        rmse = 0.0
    if a1 != a1:
        a1=0.0
    if a2 != a2:
        a2=0.0
    if a3 != a3:
        a3=0.0
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    mae = (np.abs(gt-pred)).mean()
    if abs_rel != abs_rel:
        abs_rel=0.0
    if log_10 != log_10:
        log_10=0.0
    if mae != mae:
        mae=0.0
    
    return abs_rel, rmse, a1, a2, a3, log_10, mae

class TextWrite(object):
    ''' Wrting the values to a text file 
    '''
    def __init__(self, filename):
        self.filename = filename
        self.file = open(self.filename, "w+")
        self.file.close()
        self.str_write = ''
    
    def add_line_csv(self, data_list):
        str_tmp = []
        for item in data_list:
            if isinstance(item, int):
                str_tmp.append("{:03d}".format(item))
            if isinstance(item, str):
                str_tmp.append(item)
            if isinstance(item, float):
                str_tmp.append("{:.6f}".format(item))
        
        self.str_write = ",".join(str_tmp) + "\n"
    
    def add_line_txt(self, content, size=None, maxLength = 10, heading=False):
        if size == None:
            size = [1 for i in range(len(content))]
        if heading:    
            str_tmp = '|'.join(list(map(lambda x,s:x.center((s*maxLength)+(s-1)), content, size)))
        else:
            str_tmp = '|'.join(list(map(lambda x,s:x.rjust((s*maxLength)+(s-1)), content, size)))
        self.str_write += str_tmp + "\n" 

    def write_line(self):  
        self.file = open(self.filename, "a")
        self.file.write(self.str_write)
        self.file.close()
        self.str_write = ''

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def inverse_stft_3path_attention(spec):
    s = spec.cpu().detach().numpy()
    #s = np.exp((((s + 1) * 0.5) * 32) - 18.5) - 10**-8 # Empirical (average) min and max over test set
    #s = np.exp((s * (3 + 18.42)) - 18.42) - 10**-8
    #s = np.exp((s * (1.29 + 15.21)) - 15.21) - 10**-8
    #s = np.exp(s) - 10**-8
    rp = np.random.uniform(-np.pi, np.pi, s.shape)
    f = s * (np.cos(rp) + (1.j * np.sin(rp)))
    y = librosa.istft(f) # Reconstruct audio
    return y/np.abs(y).max()

"""
train: max avg:1.2938154406513762, min avg:-15.214474225912982
val: max avg:1.3373957109108245, min avg:-15.046736742952744
test: max avg:1.2942348212166561, min avg:-15.214038060011108
"""

def inverse_normalize_stft(spec):
    s = spec.cpu().detach().numpy()
    #s = np.exp((((s + 1) * 0.5) * 32) - 18.5) - 10**-8 # Empirical (average) min and max over test set
    #s = np.exp((((s + 1) * 0.5) * (4.288 + 18.42)) - 18.42) - 10**-8
    #s = np.exp((s * (1.29 + 15.21)) - 15.21) - 10**-8
    #s = np.exp(s) - 10**-8
    return s #s/np.abs(s).max()

def compare_t60(a, b, sr=86):
    try:
        a = a.detach().clone().abs()
        b = b.detach().clone().abs()
        a = (a - a.min())/(a.max() - a.min())
        b = (b - b.min())/(b.max() - b.min())
        t_a = estimate_t60(a, sr)
        t_b = estimate_t60(b, sr)
        return abs((t_b - t_a)/t_a) * 100
    except Exception as error:
        return 100
    
def estimate_t60(audio, sr):
    fs = float(sr)
    #audio = audio.detach().clone()
    audio = audio

    decay_db = 20

    # The power of the impulse response in dB
    power = audio ** 2
    energy = torch.flip(torch.cumsum(torch.flip(power, [0]), 0), [0])  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = torch.max(torch.where(energy > 0)[0])
    n = energy[:i_nz]
    db = 10 * torch.log10(n)
    db = db - db[0]

    # -5 dB headroom
    i_5db = torch.min(torch.where(-5 - db > 0)[0])
    e_5db = db[i_5db]
    t_5db = i_5db / fs

    # after decay
    i_decay = torch.min(torch.where(-5 - decay_db - db > 0)[0])
    t_decay = i_decay / fs

    # compute the decay time
    decay_time = t_decay - t_5db
    est_rt60 = (60 / decay_db) * decay_time

    return est_rt60

HOP_LENGTH = 62 
N_FFT = 512
WIN_LENGTH = 248 

def visualization(pred_queryIR,gt_queryImpEchoes_mag, attention,opt,epoch,mode="train"):
    ROOT_PATH = opt.checkpoints_dir + "/img"
    if not os.path.exists(ROOT_PATH):
        os.mkdir(ROOT_PATH)
        print("img path {} made.".format(ROOT_PATH))

    fig, ax = plt.subplots(1,2)
    wav_pred = inverse_stft_3path_attention(pred_queryIR[0,0,:,:])
    wav_gt = inverse_stft_3path_attention(gt_queryImpEchoes_mag[0,0,:,:])
        
    #plt.figure()
    librosa.display.waveshow(wav_pred, sr=22050*2, color='orange',ax=ax[0])
    plt.legend('pred wav')
    #plt.xlim(0, DURATION)
    #plt.savefig(ROOT_PATH + "img/pred_waveform_transformer.jpg")

    #plt.figure()
    librosa.display.waveshow(wav_gt, sr=22050*2, color='blue',ax=ax[1])
    plt.legend('gt wav')
    #plt.xlim(0, DURATION)
    plt.savefig(ROOT_PATH + "/gt_and_pred_waveform_{}.jpg".format(mode))
    print("waveform file saved.")

    """
    fig, ax = plt.subplots(1,2)                                               
    ax[0].set_title('Power spectrogram pred train epoch:{}'.format(epoch + 1))
    img_pred = librosa.display.specshow(librosa.amplitude_to_db(inverse_normalize_stft(pred_queryIR[0,0,:,:])),
                                y_axis='mel', x_axis='time', ax=ax[0])
    fig.colorbar(img_pred, ax=ax[0], format="%+2.0f dB")
    #fig.colorbar(format="%+2.0f dB")
    # plt.savefig('/home/xiaohu/workspace/my_habitat/img/pred_IR_train{}.png')

    # fig2, ax2 = plt.subplots()
    # S2 = inverse_normalize_stft(gt_queryImpEchoes_mag[0,0,:,:,0])
    img_gt = librosa.display.specshow(librosa.amplitude_to_db(inverse_normalize_stft(gt_queryImpEchoes_mag[0,0,:,:])),
                                y_axis='mel', x_axis='time', ax=ax[1])
    ax[1].set_title('Power spectrogram gt train epoch:{}'.format(epoch + 1))
    fig.colorbar(img_gt, ax=ax[1], format="%+2.0f dB")
    plt.savefig(ROOT_PATH + '/IR_{}.jpg'.format(mode))
    """
    fig, ax = plt.subplots(1,2)                                               
    ax[0].set_title('Power spectrogram pred train epoch:{}'.format(epoch + 1))
    img_left = librosa.display.specshow(librosa.amplitude_to_db(inverse_normalize_stft(pred_queryIR[0,0,:,:])), n_fft = N_FFT, win_length = WIN_LENGTH, hop_length = HOP_LENGTH,
                                x_axis = 'time', y_axis = 'mel', ax=ax[0], ) #sr = 44100, n_fft = NFFT, win_length = winl)
    fig.colorbar(img_left, ax=ax[0], format="%+2.0f dB")
    # plt.savefig('/home/xiaohu/workspace/my_habitat/img/pred_IR_train{}.png')

    ax[1].set_title('Power spectrogram gt train epoch:{}'.format(epoch + 1))
    img_right = librosa.display.specshow(librosa.amplitude_to_db(inverse_normalize_stft(gt_queryImpEchoes_mag[0,0,:,:])),  n_fft = N_FFT, win_length = WIN_LENGTH, hop_length = HOP_LENGTH,
                                 x_axis = 'time', y_axis = 'mel',ax=ax[1], )#sr = 44100, n_fft = NFFT, win_length = winl)
    fig.colorbar(img_right, ax=ax[1], format="%+2.0f dB")
    plt.savefig(ROOT_PATH + '/IR_{}.jpg'.format(mode))

    plt.figure()
    attention_np = attention[0,0,:,:].cpu().detach().numpy()
    plt.imshow(np.array(attention_np/attention_np.max()*255, dtype=np.uint8), cmap='gray')
    plt.title("attention map")
    plt.savefig(ROOT_PATH + '/Attention map_{}.jpg'.format(mode))
    
    t60_error = compare_t60(torch.from_numpy(wav_gt), torch.from_numpy(wav_pred), sr=22050*2)
    return t60_error

