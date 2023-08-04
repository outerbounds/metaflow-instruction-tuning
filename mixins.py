from metaflow import Parameter, IncludeFile, JSONType
from config import load_config, TrainConfig
from gpu_profile import GPUProfiler
"""
Returns path for a file. 
"""
import tempfile

N_GPU = 2
visible_devices = str(list(range(N_GPU)))[1:-1]


def _to_file(file_bytes, extension=None):
    params = {
        "suffix": f".{extension.replace('.', '')}" if extension is not None else None,
        "delete": True,
        "dir": "./"
    }
    latent_temp = tempfile.NamedTemporaryFile(**params)
    latent_temp.write(file_bytes)
    latent_temp.seek(0)
    return latent_temp

class ConfigBase:
    """
    Base class for all config needed for this flow as well as any dependent flows.

    This class can be inherited by downstream classes or even used a mixin.
    
    This class is meant for reuse in Metaflow flows which want to resue the configuration parameters of this training flow so 
    that they can call downstream flows with the same configuration parameters.

    Example: 
    --------
    - Upstream flow which is preparing data is inheriting the configuration schema / parameters from this class
    - This way correct configuration parsed in both flows while we can also pass the configuration from the upstream flow to the downstream flow while ensuring that the configuration is valid.
    - This pattern is very useful when we have a complex configuration schema and we want to reuse it in multiple flows. These flows may be invoked asynchronously using event handlers, so having a common configuration schema parser is very useful.
    """

    def _resolve_config(self):
        if self.experiment_config is not None and self.experiment_config_file is not None:
            raise ValueError("Cannot specify both --config or --config-file")
        elif self.experiment_config is None and self.experiment_config_file is None:
            raise ValueError("Must specify either --config or --config-file")
        if self.experiment_config is not None:
            return load_config(self.experiment_config)
        if self.experiment_config_file is not None:
            temf = _to_file(bytes(self.experiment_config_file, "utf-8"),)
            return load_config(temf.name)

    _config = None

    @property
    def config(self) -> TrainConfig:
        if self._config is not None:
            return self._config
        self._config = self._resolve_config()
        return self._config

    experiment_config_file = IncludeFile("config-file", help="experiment config file", default=None)

    experiment_config = Parameter("config", help="experiment config", default=None, type=JSONType)

    def config_report(self):
        from metaflow.cards import Markdown
        from omegaconf import OmegaConf
        return [Markdown(
            f"## Experiment Config"
        ),
        Markdown(
            f"```\n{OmegaConf.to_yaml(self.config)}```"
        )]
    
class HuggingFaceLora(ConfigBase):

    def download_model_from_huggingface(self, save_dir):
        import huggingface_hub 
        from glob import glob
        import os
        try:
            huggingface_hub.snapshot_download(
                repo_id=self.config.model.base_model,
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
            model_directory = self.config.training.model_save_directory

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
        from omegaconf import OmegaConf
        import json
        base_model_path = base_model_path if base_model_path is not None else self.config.model.base_model
        data_path = dataset_path if dataset_path is not None else self.config.dataset.dataset_path
        sample_arg = []
        if self.config.dataset.num_data_samples is not None:
            sample_arg = [f"--num-samples={self.config.dataset.num_data_samples}"]
        # TODO set `--nproc_per_node` based on `visible_devices` setting.
        visible_devices = self.config.training.visible_devices
        if visible_devices is None or visible_devices == "auto":
            device_list = GPUProfiler._read_devices()
            visible_devices = str(len(device_list))
        
        prompt_dict = OmegaConf.to_container(self.config.dataset.prompt_template)
        tmpfile = _to_file(bytes(json.dumps(prompt_dict), "utf-8"), extension="json")
        import os
        subprocess.run(
            [ 
                f"torchrun",
                f"--nproc_per_node={visible_devices}",
                f"--master_port={self.config.training.master_port}",
                "tuner.py",
                f"--base_model='{base_model_path}'",
                f"--num_epochs={self.config.training.num_epochs}",
                f"--cutoff_len={self.config.training.cutoff_len}",
                f"--batch_size={self.config.training.macro_batch_size}",
                "--train_on_inputs",
                f"--output_dir={self.config.training.model_save_directory}",
                f"--lora_target_modules={self.config.lora.target_modules}",
                f"--lora_r={self.config.lora.r}",
                f"--micro_batch_size={self.config.training.micro_batch_size}",
                f"--fp16={self.config.training.fp16}",
                f"--data-path={data_path}",
                f"--prompt-template-name={str(os.path.basename(tmpfile.name)).replace('.json', '')}",
            ] + sample_arg,
            env=env,
            check=True
        )
