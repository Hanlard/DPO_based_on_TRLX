import json
import os
import uuid
from time import time
from typing import Callable, List

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.dpo_types import DPORLBatch, DPORLElement
from trlx.models.modeling_ppo import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    FixedKLController,
)

from trlx.pipeline.dpo_pipeline import DPOPipeline
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer

from trlx.utils.modeling import RunningMoments, gather_dict, logprobs_of_labels

from trlx.utils.modeling import flatten_dict

logger = logging.get_logger(__name__)

def logprobs_of_labels_fp32(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits."""
    logprobs = F.log_softmax(logits.float(), dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).half()
    return logprobs_labels.squeeze(-1)

@register_trainer
class AccelerateDPOTrainer(AccelerateRLTrainer):
    """DPO Accelerate Trainer"""
    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        self.config = config

    def get_arch(self, config: TRLConfig):
        """Get the model"""
        model_class = AutoModelForCausalLMWithHydraValueHead
        if config.model.model_arch_type == "seq2seq":
            model_class = AutoModelForSeq2SeqLMWithHydraValueHead

        from_fn = model_class.from_pretrained
        # backward-compat: Try to create a randomly initialized architecture from a config
        if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
            from_fn = model_class.from_config

        return from_fn(
            config.model.model_path,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
        )

    def loss(self, batch):
        """
        batch:  bs*chosen + bs*reject
        """
        # policy model forward()
        outputs = self.model(batch['input_ids'], batch['attention_mask'], return_dict=True)
        logits = outputs.logits

        # reference model forward()
        with torch.no_grad():
            if hasattr(self.model, "frozen_head"):
                ref_logits = self.model.forward_hydra(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_dict=True,
                ).logits
            else:
                ref_logits = self.ref_model(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_dict=True,
                ).logits
        ref_logits = ref_logits.to(batch['input_ids'].device)

        # compute the log probablity of all label tokens
        logprobs = logprobs_of_labels(logits[:, :-1, :], batch['input_ids'][:, 1:])
        ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], batch['input_ids'][:, 1:])



        # paddding_side: left
        starts = batch['s_res'] - 1
        end = -1

        n_samples = batch['input_ids'].shape[0]
        n_queries = n_samples // 2
        assert n_samples == n_queries*2, f"not equal: n_samples={n_samples},n_queries={n_queries}"

        # [2 * n_queries ,] This is pi(y|x)

        logprobs_response = []
        ref_logprobs_response = []

        for ix in range(n_samples):
            logp = logprobs[ix, starts[ix]: end].mean().unsqueeze(dim=0)
            rlogp = ref_logprobs[ix, starts[ix]: end].mean().unsqueeze(dim=0)
            logprobs_response.append(logp)
            ref_logprobs_response.append(rlogp)
            # print(ix, starts[ix], end, logprobs.shape, batch['input_ids'].shape, logits.shape ,logprobs[ix, starts[ix]: end], logprobs[ix, starts[ix]: end].mean(), logp, rlogp )


        logprobs_response = torch.cat(logprobs_response).to(batch['input_ids'].device)
        ref_logprobs_response = torch.cat(ref_logprobs_response).to(batch['input_ids'].device)

        # [n_queries, ] log (pi(y|x)/pi_ref(y|x)) = log pi(y|x) - log pi_ref(y|x)
        ratio_w =  logprobs_response[:n_queries] - ref_logprobs_response[:n_queries]
        ratio_l =  logprobs_response[n_queries:] - ref_logprobs_response[n_queries:]



        # hyper-parameter
        beta = 0.1
        loss = - torch.sigmoid(beta*ratio_w - beta*ratio_l).mean()

        # logger.info(f"loss = {loss.item()},"
        #             f"ratio_w = {ratio_w.mean().item()},"
        #             f"ratio_l = {ratio_l.mean().item()},"
        #             f"nan_logprobs = {torch.isnan(logprobs).all()},"
        #             f"nan_logprobs_response = {torch.isnan(logprobs_response).all()},"
        #             f"nan_logits = {torch.isnan(logits).all()},"
        #             f"nan_ref_logprobs = {torch.isnan(ref_logprobs).all()},"
        #             f"nan_ref_logprobs_response = {torch.isnan(ref_logprobs_response).all()},"
        #             f"nan_ref_logits = {torch.isnan(ref_logits).all()},"
        #             )

        stats = dict(
            loss=loss.item(),
            ratio_w=ratio_w.mean().item(),
            ratio_l=ratio_l.mean().item(),
            logprobs_w=logprobs_response[:n_queries].mean().item(),
            ref_logprobs_w=ref_logprobs_response[:n_queries].mean().item(),
            logprobs_l=logprobs_response[n_queries:].mean().item(),
            ref_logprobs_l=ref_logprobs_response[n_queries:].mean().item(),

        )
        return loss, stats

    def prepare_learning(self):
        train_dataloader = self.store.create_loader(self.config.train.batch_size)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        (
            self.model,
            self.opt,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(self.model, self.opt, train_dataloader, eval_dataloader)

        self.n_updates_per_batch = 1
        self.total_steps = self.config.train.epochs * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def make_experience(self, samples, input_length, output_length):

        self.store = DPOPipeline(samples, input_length, output_length, self.tokenizer)

