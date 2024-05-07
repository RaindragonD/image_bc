#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=deepaklong
#SBATCH --nodelist=grogu-0-19
#SBATCH --error=/grogu/user/rmendonc/slurm_logs/alfred.err
#SBATCH --output=/grogu/user/rmendonc/slurm_logs/alfred.out


cd /home/rmendonc/research/alfred/image_bc
source activate diffusion


CUDA_VISIBLE_DEVICES=0 python finetune.py agent=diffusion_unet exp_name=grasp_right_ft_abs_vit_pretrained \
agent.features.restore_path=/grogu/user/rmendonc/data4robotics_features/visual_features/vit_base/SOUP_1M_DH.pth \
buffer_path=/grogu/user/rmendonc/alfred_data/grasp_right/absolute/buf.pkl \
max_iterations=500000 trainer=bc_cos_sched ac_chunk=16  \
task=single_hand task.train_buffer.cam_indexes=[0] \
