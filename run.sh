 CUDA_VISIBLE_DEVICES=1 nohup /home/xiaohu/miniconda3/envs/habitatEnv/bin/python  train.py \
--validation_on \
--dataset replica \
--checkpoints_dir /home/xiaohu/workspace/beyond-depth-to-echos/results_new_new_winlength248/redo_ablation_cat3feats \
--batchSize 256 \
--img_path /home/xiaohu/workspace/data/scene_observations_128.pkl \
--metadatapath /sas1/Sascha/SharedDatasets/sound-spaces/data/metadata/replica \
--audio_path /home/xiaohu/workspace/data/echoes_navigable \
--init_material_weight /home/xiaohu/workspace/beyond-depth-to-echos/checkpoints/material_pre_trained_minc.pth \
--epoch_save_freq 500 \
--display_freq 40 \
--validation_freq 20 \
&> /home/xiaohu/workspace/beyond-depth-to-echos/python_log.txt