from metaflow import FlowSpec, step, Parameter, resources, environment, kubernetes, current, card, project, trigger, S3
from mixins import HuggingFaceLora, N_GPU, visible_devices
from custom_decorators import pip, gpu_profile
import os
from model_store import ModelStore, ModelStoreParams

# HF_IMAGE = "006988687827.dkr.ecr.us-west-2.amazonaws.com/llm/hf-lora-pt:latest"
HF_IMAGE =  "valayob/hf-transformer-gpu:4.29.2.3"

@project(name="lora")
@trigger(event="alpaca.dataprep")
class LlamaInstructionTuning(FlowSpec, HuggingFaceLora, ModelStoreParams):

    s3_dataset_path = Parameter(
        "s3-dataset-path",
        help="S3 path to the dataset; If it is not provided, then the path used in the configuration file will be used. It accepts comma separated s3 paths.",
        default=None
    )

    @card
    @kubernetes(image=HF_IMAGE, cpu=2, memory=5000)
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

    def _download_dataset_from_s3(self, tempfile_path):
        import shutil
        with S3() as s3:
            s3_resp = s3.get(self.s3_dataset_path)
            shutil.move(s3_resp.path, tempfile_path)

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
        
        dataset_temp_file = None
        if self.s3_dataset_path is not None:
            dataset_temp_file = tempfile.NamedTemporaryFile(suffix=".json")
            self._download_dataset_from_s3(dataset_temp_file.name)
            print(f"Using dataset path {self.s3_dataset_path} provided by the user dowloaded at : {dataset_temp_file}")

        if not hf_model_store.already_exists(base_model):
            raise ValueError(f"Model {base_model} not found in the model store. This shouldn't happen.")
        with tempfile.TemporaryDirectory() as datasetdir:
            with S3() as s3:
                s3.get()
                    
        with tempfile.TemporaryDirectory() as tmpdirname:
            hf_model_store.download(base_model, tmpdirname)
            self.run(
                base_model_path=tmpdirname,
                dataset_path=dataset_temp_file.name if dataset_temp_file is not None else None,
            )
            trained_model_store.upload(model_save_dir, base_model)
        
        current.card.extend(self.config_report())
        self.next(self.end)

    @step
    def end(self):
        print("Completed!")        

if __name__ == "__main__":
    LlamaInstructionTuning()
