from metaflow import Parameter

N_GPU = 2
visible_devices = str(list(range(N_GPU)))[1:-1]


class HuggingFaceLora:

    base_model = Parameter("base-model", help="model", default="yahma/llama-7b-hf")
    num_epochs = Parameter("epochs", help="number of epochs", default=1)
    macro_batch_size = Parameter(
        "macro-batch-size", help="macro batch size", default=8
    )
    visible_devices = Parameter(
        "devices", help="visible devices", default=N_GPU, type=str
    )
    world_size = Parameter("world-size", help="world size", default=N_GPU, type=str)
    cutoff_len = Parameter("cutoff", help="cutoff length", default=258)
    micro_batch_size = Parameter("micro-batch-size", help="micro batch size", default=4)
    model_save_directory = Parameter(
        "output-directory", help="model save directory", default="./lora-alpaca"
    )
    master_port = Parameter("master-port", help="master port", default=1234)
    lora_r = Parameter("r", help="lora r", default=2)
    lora_target_modules = Parameter(
        "target-modules",
        help="target lora modules you want to fine tune",
        default="[q_proj,v_proj]",
    )
    fp16 = Parameter("fp16", help="Whether to use fp16", default=True, type=bool)

    num_data_samples = Parameter("num-data-samples", help="number of data samples", default=None, type=int)

    dataset_path = Parameter(
        "dataset-path",
        help="path to the dataset; Can be a huggingface dataset or a local path",
        default="yahma/alpaca-cleaned",
    )

    def download_model_from_huggingface(self, save_dir):
        import huggingface_hub 
        from glob import glob
        import os
        try:
            huggingface_hub.snapshot_download(
                repo_id=self.base_model,
                local_dir=save_dir,
            )
            pass
        except Exception as e:
            print(e)
            raise e

    def upload_to_huggingface(self, model_directory=None):
        from huggingface_hub import HfApi
        from glob import glob
        import os
        if model_directory is None:
            model_directory = self.model_save_directory

        api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
        hf_organization = os.environ["HF_ORGANIZATION"]
        # we first create the repo on huggingface for all the checkpoints
        for checkpoint_folder in glob(f"{model_directory}/*/"):
            repo_name = checkpoint_folder.replace("_", "-").replace("/", "-")
            try:
                api.create_repo(f"{hf_organization}/{repo_name}", private=True)
                pass
            except Exception as e:
                print(e)
                pass

        # then we upload the files. We upload only what's needed for the PEFT config
        for checkpoint_folder in glob(model_directory + "/*/"):
            repo_name = checkpoint_folder.replace("_", "-").replace("/", "-")

            for ff in [
                f"{checkpoint_folder}/adapter_config.json",
                f"{checkpoint_folder}/adapter_model.bin",
                f"{checkpoint_folder}/trainer_state.json",
            ]:
                api.upload_file(
                    path_or_fileobj=ff,
                    path_in_repo=ff.split("/")[-1],
                    repo_id=f"{hf_organization}/{repo_name}",
                    repo_type="model",
                )

    def run(self, base_model_path=None, dataset_path=None, env=None):
        import subprocess
        base_model_path = base_model_path if base_model_path is not None else self.base_model
        data_path = dataset_path if dataset_path is not None else self.dataset_path
        sample_arg = []
        if self.num_data_samples is not None:
            sample_arg = [f"--num-samples={self.num_data_samples}"]
        subprocess.run(
            [ 
                f"torchrun",
                f"--nproc_per_node={self.visible_devices}",
                f"--master_port={self.master_port}",
                "tuner.py",
                f"--base_model='{base_model_path}'",
                f"--num_epochs={self.num_epochs}",
                f"--cutoff_len={self.cutoff_len}",
                f"--batch_size={self.macro_batch_size}",
                "--train_on_inputs",
                f"--output_dir={self.model_save_directory}",
                f"--lora_target_modules={self.lora_target_modules}",
                f"--lora_r={self.lora_r}",
                f"--micro_batch_size={self.micro_batch_size}",
                f"--fp16={self.fp16}",
                f"--data-path={data_path}"
            ] + sample_arg,
            env=env,
        )
