import os
import time
import torch
from options.train_options import TrainOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader
from util.util import TextWrite, compute_errors, visualization, inverse_stft_3path_attention, compare_t60
import numpy as np
from models import criterion 
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import librosa
import matplotlib.pyplot as plt
# First get the amount of CPUs
import multiprocessing
n_processors = multiprocessing.cpu_count()

# Then set the amount of THREADS a python program uses:
os.environ["OMP_NUM_THREADS"] = str(4)


def create_optimizer(nets, opt):
	(net_rgbspec, net_depthspec, net_attention, net_material) = nets
	param_groups = [{'params': net_rgbspec.parameters(), 'lr': opt.lr_visual},
					{'params': net_depthspec.parameters(), 'lr': opt.lr_audio},
					{'params': net_attention.parameters(), 'lr': opt.lr_attention},
					{'params': net_material.parameters(), 'lr': opt.lr_material},
					#{'params': net_rgbmaterial.parameters(), 'lr': opt.lr_attention},
					]
	if opt.optimizer == 'sgd':
		return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
	elif opt.optimizer == 'adam':
		return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.94):
	for param_group in optimizer.param_groups:
		param_group['lr'] *= decay_factor

def evaluate(model, loss_criterion, dataset_val, opt, epoch):
	losses = []
	errors = []
	t60_error_list = []
	with torch.no_grad():
		for i, val_data in enumerate(dataset_val):
			for key in val_data.keys():
				val_data[key] = val_data[key].cuda()
			output = model.forward(val_data)
			depth_predicted = output['spec_predicted']
			depth_gt = output['spec_gt']
   
			wav_pred = inverse_stft_3path_attention(depth_predicted[0,0,:,:])
			wav_gt = inverse_stft_3path_attention(depth_gt[0,0,:,:])
			t60_error = compare_t60(torch.from_numpy(wav_gt), torch.from_numpy(wav_pred), sr=22050*2)
   
			#t60_error = visualization(depth_predicted,depth_gt, output['attention'] ,opt,epoch,mode="validation")
			t60_error_list.append(t60_error)
     
			loss = loss_criterion(depth_predicted[depth_gt!=0], depth_gt[depth_gt!=0])
			losses.append(loss.item())
			for idx in range(depth_predicted.shape[0]):
				errors.append(compute_errors(depth_gt[idx].cpu().numpy(), 
								depth_predicted[idx].cpu().numpy()))
	
	t60_error = visualization(depth_predicted,depth_gt, output['attention'] ,opt,epoch,mode="validation")
	mean_loss = sum(losses)/len(losses)
	mean_errors = np.array(errors).mean(0)	
	mean_t60_loss  = sum(t60_error_list)/len(t60_error_list)
	print('Loss: {:.3f}, L1: {:.3f}'.format(mean_loss, mean_errors[1])) 
	val_errors = {}
	val_errors['ABS_REL'], val_errors['L1'] = mean_errors[0], mean_errors[1]
	val_errors['DELTA1'] = mean_errors[2] 
	val_errors['DELTA2'] = mean_errors[3]
	val_errors['DELTA3'] = mean_errors[4]
 
	return mean_loss, val_errors, mean_t60_loss

loss_criterion = criterion.L1Loss() #criterion.LogDepthLoss()
opt = TrainOptions().parse()
opt.device = torch.device("cuda")

exp_name = opt.checkpoints_dir.split("/")[-1]
current_time = time.strftime('%Y-%m-%d', time.localtime())
writer_path =  opt.checkpoints_dir + '/writers/'+current_time+exp_name
if not os.path.exists(writer_path):
    os.makedirs(writer_path)
writer = SummaryWriter(writer_path)

#### Log the results
loss_list = ['step', 'loss']
err_list = ['step', 'L1', 'ABS_REL', 'DELTA1', 'DELTA2', 'DELTA3']

train_loss_file = TextWrite(os.path.join(opt.expr_dir, 'train_loss.csv'))
train_loss_file.add_line_csv(loss_list)
train_loss_file.write_line()

val_loss_file = TextWrite(os.path.join(opt.expr_dir, 'val_loss.csv'))
val_loss_file.add_line_csv(loss_list)
val_loss_file.write_line()

val_error_file = TextWrite(os.path.join(opt.expr_dir, 'val_error.csv'))
val_error_file.add_line_csv(err_list)
val_error_file.write_line()
################

# network builders
builder = ModelBuilder()
#net_audiodepth = builder.build_audiodepth()
#net_rgbdepth = builder.build_rgbdepth()
#net_attention = builder.build_attention()
net_rgbspec = builder.build_rgbspec()

net_attention = builder.build_attention_transformer_encoder_spec()
#net_attention = builder.build_attention_spec()
#net_attention = builder.build_attention_spec()
#net_attention = builder.build_attention_decoder()
net_depthspec = builder.build_depthspec()


#net_attention_rgbspec = builder.build_attentionRGBSpecNet()
#net_attention_depthspec = builder.build_attentionDepthSpecNet()

#net_rgbmaterial = builder.build_rgbmaterialspec()
net_material = builder.build_material_property(init_weights=opt.init_material_weight)
# exit()
nets = (net_rgbspec, net_depthspec, net_attention, net_material)

# construct our audio-visual model
model = AudioVisualModel(nets, opt).cuda()
#model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
#model.to(opt.device)

dataloader = CustomDatasetDataLoader()
dataloader.initialize(opt)
dataset = dataloader.load_data()
dataset_size = len(dataloader)
print('#training clips = %d' % dataset_size)

if opt.validation_on:
	opt.mode = 'val'
	dataloader_val = CustomDatasetDataLoader()
	dataloader_val.initialize(opt)
	dataset_val = dataloader_val.load_data()
	dataset_size_val = len(dataloader_val)
	print('#validation clips = %d' % dataset_size_val)
	opt.mode = 'train'

optimizer = create_optimizer(nets, opt)

# initialization
total_steps = 0
batch_loss = []
best_rmse = float("inf")
best_loss = float("inf")
with tqdm(total=opt.niter) as pbar:
	for epoch in range(1, opt.niter+1):
		torch.cuda.synchronize()
		batch_loss = []   

		with tqdm(total=len(dataset)) as pbar1:
			for i, data in enumerate(dataset):
				
				for key in data.keys():
					data[key] = data[key].cuda()
				
				total_steps += opt.batchSize
				
				# forward pass
				model.zero_grad()
				#print("data",data)
				output = model.forward(data)
				
				# calculate loss
				depth_predicted = output['spec_predicted']
				depth_gt = output['spec_gt']
				loss = loss_criterion(depth_predicted[depth_gt!=0], depth_gt[depth_gt!=0])
				batch_loss.append(loss.item())

				# update optimizer
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
	
				pbar1.update(opt.batchSize)

				if(total_steps // opt.batchSize % opt.display_freq == 0):
					print('Display training progress at (epoch %d, steps %d)' % (epoch, total_steps // opt.batchSize))
					avg_loss = sum(batch_loss) / len(batch_loss)
					print('Average loss: %.5f' % (avg_loss))
					writer.add_scalar("Loss/train avg L1 loss", avg_loss, total_steps // opt.batchSize)
					batch_loss = []
					print('end of display \n')
					train_loss_file.add_line_csv([total_steps // opt.batchSize, avg_loss])
					train_loss_file.write_line()
     
					t60_error = visualization(depth_predicted,depth_gt, output['attention'] ,opt,epoch,mode="train")
					print('Average t60 error : %.5f' % float(t60_error))
					writer.add_scalar("Loss/train t60 error ", float(t60_error), total_steps // opt.batchSize)
					
				if(total_steps // opt.batchSize % opt.validation_freq == 0 and opt.validation_on):
					model.eval()
					opt.mode = 'val'
					print('Display validation results at (epoch %d, steps %d)' % (epoch, total_steps // opt.batchSize))
					val_loss, val_err, t60_err_avg = evaluate(model, loss_criterion, dataset_val, opt, epoch)
					writer.add_scalar("Loss/val avg L1 loss", val_loss, total_steps // opt.batchSize)
					print('end of display \n')

					print('Average t60 error validation : %.5f' % float(t60_err_avg))
					writer.add_scalar("Loss/val average t60 error ", float(t60_err_avg), total_steps // opt.batchSize)

					model.train()
					opt.mode = 'train'

					# save the model that achieves the smallest validation error
					if val_err['L1'] < best_rmse:
						best_rmse = val_err['L1']
						print('saving the best model (epoch %d) with validation L1 %.5f\n' % (epoch, val_err['L1']))
						torch.save(net_rgbspec.state_dict(), os.path.join(opt.expr_dir, 'rgbspec_'+opt.dataset+'.pth'))
						torch.save(net_depthspec.state_dict(), os.path.join(opt.expr_dir, 'depthspec_'+opt.dataset+'.pth'))
						torch.save(net_attention.state_dict(), os.path.join(opt.expr_dir, 'attention_'+opt.dataset+'.pth'))
						torch.save(net_material.state_dict(), os.path.join(opt.expr_dir, 'material_'+opt.dataset+'.pth'))

					
					#### Logging the values for the val set
					val_loss_file.add_line_csv([total_steps // opt.batchSize, val_loss])
					val_loss_file.write_line()
					
					err_list = [total_steps // opt.batchSize, \
						val_err['L1'], val_err['ABS_REL'], \
						val_err['DELTA1'], val_err['DELTA2'], val_err['DELTA3']]
					val_error_file.add_line_csv(err_list)
					val_error_file.write_line()

		if epoch % opt.epoch_save_freq == 0:
			print('saving the model at 5th epoch')
			torch.save(net_rgbspec.state_dict(), os.path.join(opt.expr_dir, 'rgbspec_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
			torch.save(net_depthspec.state_dict(), os.path.join(opt.expr_dir, 'depthspec_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
			torch.save(net_attention.state_dict(), os.path.join(opt.expr_dir, 'attention_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
			torch.save(net_material.state_dict(), os.path.join(opt.expr_dir, 'material_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))

		#decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
		if(opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0):
			decrease_learning_rate(optimizer, opt.decay_factor)
			print('decreased learning rate by ', opt.decay_factor)

		pbar.update()
