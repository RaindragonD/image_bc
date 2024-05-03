python finetune.py exp_name=debug_bimanual_arm agent.features.restore_path=/home/yl/visual_features/vit_base/SOUP_1M_DH.pth buffer_path=/home/fomo_data/test.pkl task=bimanual_arm

python finetune.py exp_name=bimanual_hand_wave agent.features.restore_path=/home/yl/visual_features/vit_base/SOUP_1M_DH.pth buffer_path=/home/fomo_data/0502_wave_test/test.pkl task=bimanual_hand

python finetune.py agent=diffusion_unet exp_name=dp_test buffer_path=/home/fomo_data/0502_wave_test/test.pkl task=bimanual_hand max_iterations=10 trainer=bc_cos_sched ac_chunk=16 train_transform=medium task.train_buffer.cam_indexes=[0] agent.features.feature_dim=256