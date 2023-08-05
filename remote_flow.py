from metaflow import FlowSpec, step, Parameter, resources, environment, kubernetes, current, card
from mixins import HuggingFaceLora, N_GPU, visible_devices
from custom_decorators import pip, gpu_profile
import os
from model_store import ModelStore, ModelStoreParams

# HF_IMAGE = "006988687827.dkr.ecr.us-west-2.amazonaws.com/llm/hf-lora-pt:latest"
HF_IMAGE =  "valayob/hf-transformer-gpu:4.29.2.3"
class LlamaInstructionTuning(FlowSpec, HuggingFaceLora, ModelStoreParams):

    @card
    @step
    def start(self):
        store = ModelStore(
            model_store_root = self.hf_models_cache_root
        )
        current.card.extend(self.config_report())
        import tempfile
        base_model = self.config.model.base_model
        if not store.already_exists(base_model):
            with tempfile.TemporaryDirectory() as tmpdirname:
                self.download_model_from_huggingface(tmpdirname)
                store.upload(tmpdirname, base_model)
        self.next(self.finetune)

    @gpu_profile(interval=1)
    @kubernetes(image=HF_IMAGE, gpu=N_GPU, cpu=16, memory=72000)
    @pip(libraries={"omegaconf":"2.3.0"})
    @card
    @step
    def finetune(self):
        hf_model_store = ModelStore(
            model_store_root=self.hf_models_cache_root
        )
        self.trained_model_path = self.runtime_models_root
        trained_model_store = ModelStore(
            model_store_root=self.trained_model_path
        )
        base_model = self.config.model.base_model
        model_save_dir = self.config.model.model_save_directory
        import os
        import tempfile
        if not hf_model_store.already_exists(base_model):
            raise ValueError(f"Model {base_model} not found in the model store. This shouldn't happen.")
        with tempfile.TemporaryDirectory() as tmpdirname:
            hf_model_store.download(base_model, tmpdirname)
            self.run(base_model_path=tmpdirname)
            trained_model_store.upload(model_save_dir, base_model)
        
        current.card.extend(self.config_report())
        self.next(self.end)

    @step
    def end(self):
        print("Completed!")        

if __name__ == "__main__":
    LlamaInstructionTuning()

# LOCAL: python flow.py run
# REMOTE: python flow.py --package-suffixes=.txt,.json run --with batch
