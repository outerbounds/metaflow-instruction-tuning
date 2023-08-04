from omegaconf import OmegaConf
from typing import Union, Optional
from dataclasses import dataclass, field

@dataclass
class ModelParams:
    base_model: str = "yahma/llama-7b-hf"

@dataclass
class TrainingParams:
    num_epochs: int = 1
    macro_batch_size: int = 8
    visible_devices: Optional[str] = None # Can be auto / or a number
    world_size: int = 2
    cutoff_len: int = 258
    micro_batch_size: int = 4
    model_save_directory: str = "./lora-alpaca"
    master_port: int = 1234
    fp16: bool = True

@dataclass
class LoraParams:
    r: int = 2
    target_modules: str = "[q_proj,v_proj]"

@dataclass
class PromptTemplate:
    description: str = "Template used by Alpaca-LoRA."
    prompt_input: str = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n"
    prompt_no_input: str = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Response:\n"
    response_split: str = "### Response:"


@dataclass
class DataParams:
    num_data_samples: Union[int, None] = None
    dataset_path: str = "yahma/alpaca-cleaned"
    prompt_template: PromptTemplate = field(default_factory=PromptTemplate)


@dataclass
class TrainConfig:
    training:TrainingParams = field(default_factory=TrainingParams)
    model:ModelParams = field(default_factory=ModelParams)
    lora:LoraParams = field(default_factory=LoraParams)
    dataset:DataParams = field(default_factory=DataParams)


def create_config(filepath):
    conf:TrainConfig = OmegaConf.structured(TrainConfig)
    OmegaConf.save(conf, filepath)

def load_config(filepath):
    conf = OmegaConf.load(filepath)
    schema:TrainConfig = OmegaConf.structured(TrainConfig)
    trainconf:TrainConfig = OmegaConf.merge(schema, conf)
    return trainconf
    

if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 3, "usage : `python config.py create example.yaml` / `python config.py load example.yaml`"
    if sys.argv[1] == "create":
        create_config(sys.argv[2])
    elif sys.argv[1] == "load":
        print(load_config(sys.argv[2]))
    else:
        raise ValueError("Invalid argument. Must be `create` or `load`")
