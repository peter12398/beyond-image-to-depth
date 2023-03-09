CUDA_VISIBLE_DEVICES=1 /home/xiaohu/miniconda3/envs/habitatEnv/bin/python test.py \
--dataset replica \
--batchSize 256 \
--checkpoints_dir /home/xiaohu/workspace/beyond-depth-to-echos/results_new_new_winlength248/redo_ablation_cat3feats/replica \
--img_path /home/xiaohu/workspace/data/scene_observations_128.pkl \
--metadatapath /sas1/Sascha/SharedDatasets/sound-spaces/data/metadata/replica \
--audio_path /home/xiaohu/workspace/data/echoes_navigable \
