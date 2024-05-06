python finetune.py exp_name=debug_bimanual_arm agent.features.restore_path=/home/yl/visual_features/vit_base/SOUP_1M_DH.pth buffer_path=/home/fomo_data/test.pkl task=bimanual_arm

python finetune.py exp_name=test_grasp_finetune agent.features.restore_path=/home/yl/image_bc/visual_features/vit_base/SOUP_1M_DH.pth buffer_path=~/fomo_data/test_grasp/test.pkl task=bimanual_hand

CUDA_VISIBLE_DEVICES=1 python finetune.py agent=diffusion_unet exp_name=test_grasp_dp buffer_path=~/fomo_data/test_grasp/test.pkl task=bimanual_hand trainer=bc_cos_sched ac_chunk=16 train_transform=medium task.train_buffer.cam_indexes=[0] agent.features.feature_dim=256


CUDA_VISIBLE_DEVICES=0 python finetune.py exp_name=grasp_right_ft_abs agent.features.restore_path=/home/yl/visual_features/vit_base/SOUP_1M_DH.pth buffer_path=~/fomo_data/grasp_right/absolute/buf.pkl task=single_hand ac_chunk=16

CUDA_VISIBLE_DEVICES=1 python finetune.py exp_name=grasp_right_ft_del agent.features.restore_path=/home/yl/visual_features/vit_base/SOUP_1M_DH.pth buffer_path=~/fomo_data/grasp_right/delta/buf.pkl task=single_hand ac_chunk=16 

CUDA_VISIBLE_DEVICES=0 python finetune.py agent=diffusion_unet exp_name=grasp_right_dp_abs buffer_path=~/fomo_data/grasp_right/absolute/buf.pkl task=single_hand trainer=bc_cos_sched ac_chunk=16 train_transform=medium task.train_buffer.cam_indexes=[0] agent.features.feature_dim=256 max_iterations=500000

CUDA_VISIBLE_DEVICES=1 python finetune.py agent=diffusion_unet exp_name=grasp_right_dp_del buffer_path=~/fomo_data/grasp_right/delta/buf.pkl task=single_hand trainer=bc_cos_sched ac_chunk=16 train_transform=medium task.train_buffer.cam_indexes=[0] agent.features.feature_dim=256 max_iterations=500000