CUDA_VISIBLE_DEVICES=1 /home/xiaohu/miniconda3/envs/habitatEnv/bin/python get_max_dataset.py \
--dataset replica \
--batchSize 1 \
--checkpoints_dir /home/xiaohu/workspace/beyond-depth-to-echos/results_new/original_3_paths/replica \
--img_path /home/xiaohu/workspace/data/scene_observations_128.pkl \
--metadatapath /sas1/Sascha/SharedDatasets/sound-spaces/data/metadata/replica \
--audio_path /home/xiaohu/workspace/data/echoes_navigable \
