from typing import Any, Dict, Iterable, List, Tuple, Union
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline
@register_datapipeline
class DPOPipeline(BasePipeline):
# class DPOPipeline(): #debug
    """
    Dataloader which is used to supply prompts for either training or evaluation

    Args:
        samples (`List[str]` or `List[Dict[str, Any]]`): list of raw text prompts or a dictionary with a required
            key `"prompt"` and extra information, that would be passed along the generation for that prompt as a
            keyword argument to a reward function.
        max_prompt_length (`int`): max length of the prompt, if exceeded the prompt will be truncated according to
            tokenizer's truncation setting.
        tokenizer (`transformers.PreTrainedTokenizer`): a tokenizer to tokenize prompts with.
        add_special_tokens (`bool`): whether to encode prompts with tokenizer's special tokens (passed directly
            into `tokenizer.encode`)
    """

    def __init__(
        self,
        samples: Union[Dict[str, Any], List[str]],
        input_length: int,
        output_length: int,
        tokenizer: PreTrainedTokenizer,
        add_special_tokens: bool = False,
    ):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length

        prompts = [sample['prompt'] for sample in samples]
        chosens = [sample['chosen'] for sample in samples]
        rejects = [sample['reject'] for sample in samples]

        query_tokens = tokenizer(
            prompts, truncation=True, padding=False, max_length=input_length, add_special_tokens=False
        )["input_ids"]
        chosen_tokens = tokenizer(
            chosens, truncation=True, padding=False, max_length=output_length, add_special_tokens=False
        )["input_ids"]
        reject_tokens = tokenizer(
            rejects, truncation=True, padding=False, max_length=output_length, add_special_tokens=False
        )["input_ids"]

        self.tokenizer = tokenizer
        self.samples = [
            {"query_tokens": query, "chosen_tokens": chosen, "reject_tokens":reject }
            for query, chosen, reject in zip(query_tokens, chosen_tokens, reject_tokens)
        ]

    def __getitem__(self, ix: int):
        return self.samples[ix]

    def __len__(self) -> int:
        return len(self.samples)

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        def collate_fn(xs):
            input_ids_c=[]
            input_ids_r=[]
            len_q = []
            len_c = []
            len_r = []
            for x in xs:
                input_ids_c.append(x['query_tokens'] + x['chosen_tokens'])
                input_ids_r.append(x['query_tokens'] + x['reject_tokens'])
                len_q.append(len(x['query_tokens']))
                len_c.append(len(x['chosen_tokens']))
                len_r.append(len(x['reject_tokens']))

            input_ids = input_ids_c + input_ids_r
            len_prompt = len_q * 2
            len_response = len_c + len_r

            max_len=max([l1+l2 for l1,l2 in zip(len_prompt,len_response)])

            # make sure: padding_side is left
            assert self.tokenizer.padding_side == "left", f"padding_side: {self.tokenizer.padding_side}"

            out = self.tokenizer.pad([{"input_ids": token,
                                       "s_pmt": max_len-len_prompt[i]-len_response[i],
                                       "s_res": max_len-len_response[i],
                                       } for i,token in enumerate(input_ids)],
                                     max_length=max_len,
                                     return_tensors="pt")

            # for key in xs[0]:
            #     if key not in ["attention_mask", ]:
            #         out[key] = [x[key] for x in xs]
            return out

        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

# if __name__ == '__main__':
#     from transformers import LlamaTokenizer
#     path = "D:\work\Research_HUB\RLHF\\trlx\examples\lm_rlhf\download\llama_zh_hf_v2"
#     tokenizer = LlamaTokenizer.from_pretrained(path)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side='left'
#     tokenizer.truncation_side='right'
#     samples = [
#         {"prompt":"q1你吃了吗？q1","chosen":"c1我吃了c1","reject":"r1没有没有没有我吃了r1"},
#         {"prompt":"q2我吃了我吃了我吃了我吃了你吃了吗？q2","chosen":"c2我吃了我吃了我吃了我吃了我吃了c2","reject":"r2没有没有没有我吃了r2"},
#         {"prompt":"q3你吃了吗？我吃了我吃了q3","chosen":"c3我吃了我吃了c3","reject":"r3没有没有没有我吃了r3"},
#         {"prompt":"q4你吃了吗？q4","chosen":"c4我吃了我吃了我吃了c4","reject":"r4没有没有没有我吃了r4"},
#     ]
#     pipe = DPOPipeline(samples=samples, input_length=32,output_length=64,tokenizer=tokenizer)
#
#     dataloader = pipe.create_loader(batch_size=2)
#     for batch in dataloader:
#         for i in range(2*2):
#             txt0 = tokenizer.decode(batch['input_ids'][i][batch['s_pmt'][i]:batch['s_res'][i]])
#             txt1 = tokenizer.decode(batch['input_ids'][i][batch['s_res'][i]:])
#             print(txt0,txt1)
# """
# q1你吃了 c1我吃了c1
# q2我吃了 c2我吃了我吃了我吃了
# q1你吃了 r1没有没有没有我吃了r
# q2我吃了 r2没有没有没有我吃了r
# q3你吃了 c3我吃了我吃了c3
# q4你吃了 c4我吃了我吃了我吃了
# q3你吃了 r3没有没有没有我吃了r
# q4你吃了 r4没有没有没有我吃了r
# """