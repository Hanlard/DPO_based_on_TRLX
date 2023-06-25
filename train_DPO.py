import os
from typing import List
import requests
import torch
# from datasets import load_dataset
from datasets import load_from_disk
from tqdm import tqdm
from transformers import LlamaTokenizer,AutoTokenizer
from deepspeed import comm as dist
import trlx
import re
import json
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig
import random



train_bs_1gpu = 2
rollout_times = 1000
num_rollouts = 16
N_ddp = 4
ppo_epochs = 1
chunk_size = 8
num_layers_unfrozen = 2

README = "DPO"
checkpoint_dir = 'save_dir/xxx'
SFT_MODEL_PATH = "download/xxx"
DATA_DIR = "data_dir/xxx"

print(f"total batch:\t{ int(train_bs_1gpu*N_ddp) }")
print(f"num_layers_unfrozen:\t{num_layers_unfrozen}")
print(f"SFT_MODEL_PATH:\t{SFT_MODEL_PATH}")
print(f"DATA_DIR:\t{DATA_DIR}")
print(README)


# ENTITY_NAME = 'Peng_Cheng_Lab'
GROUP_NAME  = 'Alignment_Chatmind_7B'

config = TRLConfig(
    train=TrainConfig(
        seq_length=512,
        epochs=rollout_times,
        total_steps=1000000,
        batch_size=train_bs_1gpu,
        checkpoint_interval=2000,
        checkpoint_dir=checkpoint_dir,
        eval_interval=200,
        pipeline="DPOPipeline",
        trainer="AccelerateDPOTrainer",
        # entity_name=ENTITY_NAME,
        group_name=GROUP_NAME
    ),
    model=ModelConfig(model_path=SFT_MODEL_PATH,num_layers_unfrozen=num_layers_unfrozen),
    tokenizer=TokenizerConfig(tokenizer_path=SFT_MODEL_PATH,truncation_side="right",padding_side='left'),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
    method=PPOConfig(
        name="DPOConfig",
        num_rollouts=num_rollouts,
        chunk_size=chunk_size,
        ppo_epochs=ppo_epochs,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10.0,
        gen_kwargs=dict(
            max_new_tokens=256,
            top_k=20,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)

def Steam_large_score(prompts=None,predictions=None,PyTorch_REST_API_URL = 'http://192.168.242.67:8124/predict'):
    """
    8124: XL
    8123: Large
    """
    payload = {'prompts': prompts,
               "predictions": predictions,
               }
    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, data=payload).json()
    if r['success']:
        score = r['score']
        return score
    else:
        print(r)
        return None


def get_scores(prompts=None,predictions=None):
    scores_list =Steam_large_score(prompts,predictions)
    scores = torch.tensor(scores_list)
    return scores

def reward_fn(samples: List[str],prompts=None,outputs=None, **kwargs):
    #     if torch.distributed.get_rank() == 0:
    #         logging.info(f"Line 86 sample-0: {samples[0]}") # ...<text>... TL;DR: <text>
    prompts = [text for text in prompts]
    predictions = [text for text in outputs]
    scores = get_scores(prompts, predictions)
    return scores

def get_samples(data_dir):
    json_list = os.listdir(data_dir)
    json_list = [f for f in json_list if f.endswith(".json")]
    all_data = []
    for file in json_list:
        with open(f"{data_dir}/{file}", "r", encoding="utf-8") as f:
            data_list = json.loads(f.read())
            all_data.extend(data_list)
    def construct_pairs(sample):
        item_list = []
        prompt = sample['prompt']
        ans_list = sample['answers']
        n = len(ans_list)

        for i in range(n):
            for j in range(i + 1, n):
                if ans_list[i]['score'] > ans_list[j]['score']:
                    item = {"prompt": prompt,
                            "chosen": ans_list[i]['answer'],
                            "reject": ans_list[j]['answer'],
                            "score_c": ans_list[i]['score'],
                            "score_r": ans_list[j]['score'],
                            }
                    item_list.append(item)
        return item_list
    all_pairs = []
    for sample in all_data:
        pair_list = construct_pairs(sample)
        all_pairs.extend(pair_list)
    random.seed(666)
    random.shuffle(all_pairs)
    n = len(all_pairs)
    n1 = int(0.8 * n)
    n2 = int(0.9 * n)
    train = all_pairs[:n1]
    valid = all_pairs[n1:n2]
    test = all_pairs[n2:]
    return train,valid,test


if __name__ == "__main__":

    if "llama" in config.tokenizer.tokenizer_path.lower():
        tokenizer = LlamaTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    train_samples, valid_samples, _ = get_samples(DATA_DIR)

    trainer = trlx.train(
        metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)},
        samples=train_samples,
        eval_prompts=[sample['prompt'] for sample in valid_samples],
        config=config,
    )
