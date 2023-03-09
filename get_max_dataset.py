import os
import time
import torch
from options.train_options import TrainOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader
from util.util import TextWrite, compute_errors
import numpy as np
from models import criterion 
from tqdm import tqdm

def create_optimizer(nets, opt):
	(net_rgbspec, net_depthspec, net_attention, net_material) = nets
	param_groups = [{'params': net_rgbspec.parameters(), 'lr': opt.lr_visual},
                    {'params': net_depthspec.parameters(), 'lr': opt.lr_audio},
                    {'params': net_attention.parameters(), 'lr': opt.lr_attention},
                    {'params': net_material.parameters(), 'lr': opt.lr_material}
                    ]
	if opt.optimizer == 'sgd':
		return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
	elif opt.optimizer == 'adam':
		return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.94):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

def evaluate(model, loss_criterion, dataset_val, opt):
	losses = []
	errors = []
	with torch.no_grad():
		for i, val_data in enumerate(dataset_val):
			for key in val_data.keys():
				val_data[key] = val_data[key].cuda()
			output = model.forward(val_data)
			depth_predicted = output['spec_predicted']
			depth_gt = output['spec_gt']
			loss = loss_criterion(depth_predicted[depth_gt!=0], depth_gt[depth_gt!=0])
			losses.append(loss.item())
			for idx in range(depth_predicted.shape[0]):
				errors.append(compute_errors(depth_gt[idx].cpu().numpy(), 
								depth_predicted[idx].cpu().numpy()))
	
	mean_loss = sum(losses)/len(losses)
	mean_errors = np.array(errors).mean(0)	
	print('Loss: {:.3f}, RMSE: {:.3f}'.format(mean_loss, mean_errors[1])) 
	val_errors = {}
	val_errors['ABS_REL'], val_errors['RMSE'] = mean_errors[0], mean_errors[1]
	val_errors['DELTA1'] = mean_errors[2] 
	val_errors['DELTA2'] = mean_errors[3]
	val_errors['DELTA3'] = mean_errors[4]
	return mean_loss, val_errors 

loss_criterion = criterion.LogDepthLoss()
opt = TrainOptions().parse()
opt.device = torch.device("cuda")
opt.batchSize = 1
print("batch size:{}".format(opt.batchSize))

opt.mode = 'train'
dataloader_train = CustomDatasetDataLoader()
dataloader_train.initialize(opt)
dataloader_train = dataloader_train.load_data()
dataset_size_train = len(dataloader_train)
print('#train clips = %d' % dataset_size_train)

opt.mode = 'val'
dataloader_val = CustomDatasetDataLoader()
dataloader_val.initialize(opt)
dataset_val = dataloader_val.load_data()
dataset_size_val = len(dataloader_val)
print('#validation clips = %d' % dataset_size_val)

opt.mode = 'test'
dataloader_test = CustomDatasetDataLoader()
dataloader_test.initialize(opt)
dataloader_test = dataloader_test.load_data()
dataset_size_test = len(dataloader_test)
print('#test clips = %d' % dataset_size_test)

# initialization
total_steps = 0

max_list = [[],[],[]]
min_list = [[],[],[]]

with tqdm(total=3) as pbar:
	for index,dataset in enumerate([dataloader_train,dataloader_val,dataloader_test]):  

		with tqdm(total=len(dataset)) as pbar1:
			for i, data in enumerate(dataset):
			
				spec_gt = data["audio"]
				max_list[index].append(float(spec_gt.max()))
				min_list[index].append(float(spec_gt.min()))

				pbar1.update(opt.batchSize)	
		pbar.update()

print("train: max avg:{}, min avg:{}".format(np.average(np.array(max_list[0])),np.average(np.array(min_list[0]))))
print("val: max avg:{}, min avg:{}".format(np.average(np.array(max_list[1])),np.average(np.array(min_list[1]))))
print("test: max avg:{}, min avg:{}".format(np.average(np.array(max_list[2])),np.average(np.array(min_list[2]))))
