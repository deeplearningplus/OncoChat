# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from mamba import modeling_mamba

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    bfloat16: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use bfloat16 during evaluation"
        },
    )
    float16: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use float16 during evaluation"
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    output_file: str = field(default=None, metadata={"help": "Path to predicted output file"})
    device: str = field(default=None, metadata={"help": "cuda device"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    #conv = get_conversation_template("vicuna")
    conv = get_conversation_template("qwen")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    user = conv.roles[0]
    assistant = conv.roles[1]

    system_ids = tokenizer(conv.system_message, add_special_tokens=True)["input_ids"] # Add bos_token_id at the begin
    system_labels = [IGNORE_TOKEN_ID] * len(system_ids) # Do not compute loss on system prompt

    eos_token = tokenizer.eos_token
    assert eos_token is not None

    inputs = {}
    targets = {}
    attention_mask = {}

    # Apply prompt templates
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        input_ids = []
        labels = []
        input_ids += system_ids
        labels += system_labels

        # Format the multi-turn conversations as:
        ## USER: What is your name? ASSISTANT: My name is an AI assistant.<eos_token>
        # Add eos_token at the end of each conversation.
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if role == user:#"<|im_start|>user":
                message = f"{user}: {sentence['value']} {assistant}: "
                conv_ids = tokenizer(message, add_special_tokens=False)["input_ids"]
                conv_labels = [IGNORE_TOKEN_ID] * len(conv_ids) # Do not compute loss on USER input
            elif role == assistant:#"<|im_start|>assistant":
                message = f"{sentence['value']}{eos_token}"
                conv_ids = tokenizer(message, add_special_tokens=False)["input_ids"]
                conv_labels = conv_ids # Compute loss on ASSISTANT response
            else:
                raise ValueError(f"Unknown role: {role}")
            
            input_ids += conv_ids
            labels += conv_labels

            break

        #if len(input_ids) > tokenizer.model_max_length:
        #    rank0_print(
        #        f"WARNING: shorten input_ids ({len(input_ids)}) to model_max_length ({tokenizer.model_max_length})."
        #    )
        #    input_ids = input_ids[0: tokenizer.model_max_length]
        #    labels = labels[0: tokenizer.model_max_length]
        #s = tokenizer.decode(input_ids)
        #print(s)

        inputs[i] = torch.tensor(input_ids)
        targets[i] = torch.tensor(labels)
        attention_mask[i] = torch.ones_like(inputs[i])
    return dict(
        input_ids=inputs,
        labels=targets,
        attention_mask=attention_mask,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "attention_mask", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

@dataclass
class DataCollatorForLeftPadding(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "attention_mask", "labels"))
        reversed_input_ids = [seq.flip(dims=(0,)) for seq in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        input_ids = torch.stack([seq.flip(dims=(0,)) for seq in input_ids])
        return dict(input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    #import gzip
    train_json = json.load(open(data_args.data_path, "r"))
    #train_json = json.load(gzip.open(data_args.data_path, "rt"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    #data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForLeftPadding(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    assert data_args.output_file is not None

    # Set RoPE scaling factor
    #config = transformers.AutoConfig.from_pretrained(
    #    model_args.model_name_or_path,
    #    cache_dir=training_args.cache_dir,
    #    trust_remote_code=model_args.trust_remote_code,
    #)
    #orig_ctx_len = getattr(config, "max_position_embeddings", None)
    #if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
    #    scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
    #    config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    #config.use_cache = False

    # Load model and tokenizer
    #model = transformers.AutoModelForCausalLM.from_pretrained(
    model = modeling_mamba.MambaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        #config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    if model_args.bfloat16:
        model = model.bfloat16()
    elif model_args.float16:
        model = model.half()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=(model.config.model_type != 'llama'),
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    train_dataset = data_module["train_dataset"]
    device = data_args.device

    model = model.to(device)
    model.eval()

    results = []

    data_collator = data_module["data_collator"] 
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False, collate_fn=data_collator, num_workers=4)
    top_k = 5
    for inputs in tqdm(dataloader, total=len(dataloader)):
        for k, v in inputs.items():inputs[k] = v.to(device)
        #del inputs["attention_mask"]
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=100).cpu()
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #print(decoded_outputs)
        decoded_outputs = [s.split("assistant:")[1].strip() for s in decoded_outputs]

        #outputs3 = model.generate(**inputs, do_sample=True, max_new_tokens=100, num_return_sequences=3).cpu()
        #decoded_outputs3 = tokenizer.batch_decode(outputs3, skip_special_tokens=True)
        #decoded_outputs3 = [s.split("assistant:")[1].strip() for s in decoded_outputs3]

        #outputs5 = model.generate(**inputs, do_sample=True, max_new_tokens=100, num_return_sequences=5).cpu()
        #decoded_outputs5 = tokenizer.batch_decode(outputs5, skip_special_tokens=True)
        #decoded_outputs5 = [s.split("assistant:")[1].strip() for s in decoded_outputs5]

        #results += decoded_outputs

        #outs = [dict(greedy=out, top3=decoded_outputs3[i*3: (i+1)*3], top5=decoded_outputs5[i*5: (i+1)*5]) for i, out in enumerate(decoded_outputs)]
        outs = [dict(greedy=out) for i, out in enumerate(decoded_outputs)]
        results += outs
        

    #for a in tqdm(train_dataset):
    #    input_ids = a["input_ids"].to(device).unsqueeze(0)

    #    output = model.generate(input_ids, do_sample=False, max_new_tokens=100).cpu()[0]
    #    greedy_cancer_type = tokenizer.decode(output, skip_special_tokens=True).split("ASSISTANT:")[1].strip()

    #    outputs = model.generate(input_ids, do_sample=True, max_new_tokens=100, num_return_sequences=5).cpu()

    #    top_cancer_types = set()
    #    for i, output in enumerate(outputs):
    #        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
    #        cancer_type = decoded_output.split("ASSISTANT:")[1]
    #        top_cancer_types.add(cancer_type.strip())
            #print(f"Generated sequence {i+1}: {cancer_type}")

    #    results.append( {'greedy':greedy_cancer_type, 'top_5':top_cancer_types} )

    with open(data_args.output_file, "w") as fout:
        json.dump(results, fout, indent=2)

if __name__ == "__main__":
    train()
