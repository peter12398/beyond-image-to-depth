
import os 
import torch
import numpy as np
from options.test_options import TestOptions
import torchvision.transforms as transforms
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader
from util.util import compute_errors
from models import criterion 
from util.util import TextWrite, compute_errors, visualization, inverse_stft_3path_attention, compare_t60


#loss_criterion = criterion.LogDepthLoss()
loss_criterion = criterion.L1Loss()
opt = TestOptions().parse()
opt.device = torch.device("cuda")

builder = ModelBuilder()

net_depthspec = builder.build_depthspec(
                    weights=os.path.join(opt.checkpoints_dir, 'depthspec_'+opt.dataset+'.pth'))
net_rgbspec = builder.build_rgbspec(
                    weights=os.path.join(opt.checkpoints_dir, 'rgbspec_'+opt.dataset+'.pth'))

net_attention = builder.build_attention_transformer_encoder_spec(
                    weights=os.path.join(opt.checkpoints_dir, 'attention_'+opt.dataset+'.pth'))
#net_attention = builder.build_attention_spec(
#                    weights=os.path.join(opt.checkpoints_dir, 'attention_'+opt.dataset+'.pth'))
net_material = builder.build_material_property(
                    weights=os.path.join(opt.checkpoints_dir, 'material_'+opt.dataset+'.pth'))
nets = (net_rgbspec, net_depthspec, net_attention, net_material)

# construct our audio-visual model
model = AudioVisualModel(nets, opt)
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(opt.device)
model.eval()


opt.mode = 'test'
dataloader_val = CustomDatasetDataLoader()
dataloader_val.initialize(opt)
dataset_val = dataloader_val.load_data()
dataset_size_val = len(dataloader_val)
print('#validation clips = %d' % dataset_size_val)

"""
output =  {'img_spec': img_spec ,
                    'depth_spec': depth_spec ,
                    'spec_predicted': spec_prediction , 
                    'attention': alpha,
                    'img': rgb_input,
                    'spec_gt': audio_input,
                    'depth': depth_gt}
                    """

losses, errs = [], []
t60_error_list = []
with torch.no_grad():
    for i, val_data in enumerate(dataset_val):
        output = model.forward(val_data)
        depth_predicted = output['spec_predicted']
        depth_gt = output['spec_gt']
        img_depth = output['img_spec']
        audio_depth = output['depth_spec']
        attention = output['attention']
        
        wav_pred = inverse_stft_3path_attention(depth_predicted[0,0,:,:])
        wav_gt = inverse_stft_3path_attention(depth_gt[0,0,:,:])
        t60_error = compare_t60(torch.from_numpy(wav_gt), torch.from_numpy(wav_pred), sr=22050*2)

        #t60_error = visualization(depth_predicted,depth_gt, output['attention'] ,opt,epoch,mode="validation")
        t60_error_list.append(t60_error)
   
        loss = loss_criterion(depth_predicted[depth_gt!=0], depth_gt[depth_gt!=0])
        losses.append(loss.item())
        
        for idx in range(depth_gt.shape[0]):
            errs.append(compute_errors(depth_gt[idx].cpu().numpy(), 
                                depth_predicted[idx].cpu().numpy()))
            
                 
mean_loss = sum(losses)/len(losses)
mean_t60_loss = sum(t60_error_list)/len(t60_error_list)
mean_errs = np.array(errs).mean(0)

print('L1 Loss: {:.3f}, t60 Loss: {:.3f}, RMSE: {:.3f}'.format(mean_loss, mean_t60_loss, mean_errs[1])) 

errors = {}
errors['ABS_REL'], errors['RMSE'], errors['LOG10'] = mean_errs[0], mean_errs[1], mean_errs[5]
errors['DELTA1'], errors['DELTA2'], errors['DELTA3'] = mean_errs[2], mean_errs[3], mean_errs[4]
errors['MAE'] = mean_errs[6]

print('ABS_REL:{:.3f}, LOG10:{:.3f}, MAE:{:.3f}'.format(errors['ABS_REL'], errors['LOG10'], errors['MAE']))
print('DELTA1:{:.3f}, DELTA2:{:.3f}, DELTA3:{:.3f}'.format(errors['DELTA1'], errors['DELTA2'], errors['DELTA3']))
print('==='*25)
