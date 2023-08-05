from omegaconf import OmegaConf, MISSING
from typing import Union, Optional, List
from dataclasses import dataclass, field

@dataclass
class ModelParams:
    base_model: str = "yahma/llama-7b-hf"
    resuming_checkpoint_path : Optional[str] = None
    model_save_directory: str = "./lora-alpaca"
    local_model: bool = False

@dataclass
class TrainingParams:
    num_epochs: int = 1
    macro_batch_size: int = 8
    visible_devices: Optional[str] = None # Can be auto / or a number
    cutoff_len: int = 258
    micro_batch_size: int = 4
    learning_rate: float = 3e-4
    master_port: int = 1234
    fp16: bool = True
    eval_steps: int = 200
    group_by_length: bool = True
    optimizer: str = "adamw_torch"
    logging_steps: int = 100
    warmup_steps: int = 100
    

@dataclass
class LoraParams:
    r: int = 2
    target_modules: List[str] = field(default_factory=lambda: ["q_proj","v_proj"])
    alpha:int = 16
    dropout:float = 0.05
    bias:str = "none"
    task_type:str = "CAUSAL_LM"


@dataclass
class PromptTemplate:
    description: str = "Template used by Alpaca-LoRA."
    prompt_input: str = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n"
    prompt_no_input: str = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Response:\n"
    response_split: str = "### Response:"


@dataclass
class TokenizationParams:
    add_eos_token: bool = True
    cutoff_len: int = 258
    train_on_inputs: bool = True

@dataclass
class DataParams:
    num_samples: Union[int, None] = None
    huggingface_dataset_path: Optional[str] = "yahma/alpaca-cleaned"
    local_dataset_path: Optional[str] = None
    prompt_template: PromptTemplate = field(default_factory=PromptTemplate)
    tokenization: TokenizationParams = field(default_factory=TokenizationParams)
    val_set_size: int = 200
    

@dataclass
class WandbParams:
    watch:str = "all"
    project:str = MISSING
    run_name: str = MISSING
    log_model: bool = True


@dataclass
class TrainConfig:
    training:TrainingParams = field(default_factory=TrainingParams)
    model:ModelParams = field(default_factory=ModelParams)
    lora:LoraParams = field(default_factory=LoraParams)
    dataset:DataParams = field(default_factory=DataParams)
    wandb:Optional[WandbParams] = None


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
