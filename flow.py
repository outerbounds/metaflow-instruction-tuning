from metaflow import FlowSpec, step, Parameter, resources, environment
from mixins import HuggingFaceLora, N_GPU, visible_devices
from metaflow.config import DATASTORE_SYSROOT_S3
from custom_decorators import pip, gpu_profile
import os
from model_store import ModelStore

class ModelCacheParams:
    model_cache_s3_prefix = Parameter("cache-s3-prefix", help="huggingface-models", default="")

    model_cache_s3_base_path = Parameter(
        "cache-s3-base-path",
        help="By default this will use the `metaflow.metaflow_config.DATASTORE_SYSROOT_S3` ie the `METAFLOW_DATASTORE_SYSROOT_S3` configuration variable and use the path to it's parent directory. You can override this by specifying a different path here.",
        default=os.path.dirname(DATASTORE_SYSROOT_S3),
    )

    @property
    def cache_root(self):
        return os.path.join(self.model_cache_s3_base_path, self.model_cache_s3_prefix)


class LlamaInstructionTuning(FlowSpec, HuggingFaceLora, ModelCacheParams):

    push_checkpoints = Parameter(
        "push", help="push checkpoints on huggingface", default=False, type=bool
    )

    @step
    def start(self):
        store = ModelStore(
            model_store_root = self.cache_root
        )
        if not store.already_exists(self.base_model):
            self.download_model_from_huggingface()
            store.upload(self.model_save_directory, self.base_model)
        self.next(self.finetune)

    @environment(
        vars={
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "WORLD_SIZE": N_GPU,
            "HUGGINGFACE_TOKEN": os.environ["HUGGINGFACE_TOKEN"],
            "HF_ORGANIZATION": os.environ["HF_ORGANIZATION"],
        }
    )
    @gpu_profile(interval=1)
    @pip(file="requirements.txt")
    @resources(gpu=N_GPU, cpu=16, memory=128000)  # tested with A100 and A6000 GPU.
    @step
    def finetune(self):
        store = ModelStore(
            model_store_root=self.cache_root
        )
        store.download(self.base_model, self.model_save_directory)
        self.run()
        if self.push_checkpoints:
            self.upload_to_huggingface()
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    LlamaInstructionTuning()

# LOCAL: python flow.py run
# REMOTE: python flow.py --package-suffixes=.txt,.json run --with batch
