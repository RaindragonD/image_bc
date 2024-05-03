python finetune.py exp_name=debug_bimanual_arm agent.features.restore_path=/home/yl/visual_features/vit_base/SOUP_1M_DH.pth buffer_path=/home/fomo_data/test.pkl task=bimanual_arm

python finetune.py exp_name=bimanual_hand_wave agent.features.restore_path=/home/yl/visual_features/vit_base/SOUP_1M_DH.pth buffer_path=/home/fomo_data/0502_wave_test/test.pkl task=bimanual_hand