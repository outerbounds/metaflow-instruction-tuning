import os
import sys
from typing import List
import fire
import torch
import transformers
from datasets import load_dataset
import omegaconf
from config import TrainConfig, load_config
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompter import Prompter, neither_is_none_or_both_are_none, select_first_non_none


def train(config_file) -> None:
    config: TrainConfig = load_config(config_file)
    base_model: str = config.model.base_model  # the only required argument # TrainingParams.base_model
    assert (
        config.model.base_model
    ), "Please specify a model.base_model, e.g. 'huggyllama/llama-7b'"
    if neither_is_none_or_both_are_none(config.dataset.local_dataset_path, config.dataset.huggingface_dataset_path):
        raise ValueError(
            "Please specify either a dataset.local_dataset_path or a dataset.huggingface_dataset_path, but not both."
        )
    
    gradient_accumulation_steps = config.training.macro_batch_size // config.training.micro_batch_size
    
    prompter = Prompter(template_object=config.dataset.prompt_template)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        print("Training with DDP")
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if gradient_accumulation_steps < 1:
        gradient_accumulation_steps = 1

    use_wandb = False
    if config.wandb is not None:
        # Check if parameter passed or if set within environ
        use_wandb = True
        # Only overwrite environ if wandb param passed
        os.environ["WANDB_PROJECT"] = str(config.wandb.project)
        os.environ["WANDB_WATCH"] = str(config.wandb.watch)
        os.environ["WANDB_LOG_MODEL"] = str(config.wandb.log_model)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("Config:")
        print(omegaconf.OmegaConf.to_yaml(config))
        if config.model.local_model:
            print("Loading local model", base_model)
        
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True if not config.training.fp16 else False,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    
    cuttoff_len = config.dataset.tokenization.cutoff_len
    add_eos_token = config.dataset.tokenization.add_eos_token
    train_on_inputs = config.dataset.tokenization.train_on_inputs

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cuttoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cuttoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )
    model = get_peft_model(model, config)

    
    if config.dataset.local_dataset_path:
        data = load_dataset("json", data_files=config.dataset.local_dataset_path)
    else:
        data = load_dataset(config.dataset.huggingface_dataset_path)

    resuming_checkpoint_path = config.model.resuming_checkpoint_path
    if resuming_checkpoint_path:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resuming_checkpoint_path, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resuming_checkpoint_path, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resuming_checkpoint_path = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data = None
    val_set_size = config.dataset.validation_set_size
    num_samples = config.dataset.num_samples
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        if num_samples is not None and num_samples > 0:
            train_data = train_val["train"].select(range(num_samples)).shuffle().map(generate_and_tokenize_prompt)
        else:
            train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        if num_samples is not None and num_samples > 0:
            train_data = data["train"].select(range(num_samples)).shuffle().map(generate_and_tokenize_prompt)
        else:
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=config.training.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=config.training.warmup_steps,
            num_train_epochs=config.training.num_epochs,
            learning_rate=config.training.learning_rate,
            fp16=config.training.fp16,
            logging_steps=config.training.logging_steps,
            optim=config.training.optimizer,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=config.training.eval_steps if val_set_size > 0 else None,
            save_steps=200,
            output_dir=config.model.model_save_directory,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=config.training.group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=config.wandb.run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # with torch.autocast("cuda"):
    trainer.train(resume_from_checkpoint=resuming_checkpoint_path)

    model.save_pretrained(config.model.model_save_directory)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
