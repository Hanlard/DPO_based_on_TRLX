# DPO_based_on_TRLX
Reproducing the code of "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
Codes of DPO based Trlx

"""
CUDA_VISIBLE_DEVICES=$2 accelerate launch --config_file configs/dpo_accelerate_config.yaml --main_process_port $3 $1.py
"""
