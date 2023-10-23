from metaflow import FlowSpec, step, Parameter, resources, environment, kubernetes, current, card, project, trigger, S3
from mixins import HuggingFaceLora, N_GPU, visible_devices
from custom_decorators import pip, gpu_profile
import tempfile
from model_store import ModelStore, ModelStoreParams

# HF_IMAGE = "006988687827.dkr.ecr.us-west-2.amazonaws.com/llm/hf-lora-pt:latest"
HF_IMAGE =  "valayob/hf-transformer-gpu:4.29.2.4"

@project(name="lora")
@trigger(event={"name":"alpaca.dataprep", "parameters":{"s3-dataset-path":"dataset_path"}})
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
        # Cache the Huggingface model to S3 one-time for faster access
        # `ModelStore` is a utility library written for this project 
        # to help with caching and loading models from S3
        store = ModelStore(
            model_store_root = self.hf_models_cache_root
        )
        
        # If it exists in cache, then don't cache it again. 
        base_model = self.config.model.base_model
        if not store.already_exists(base_model):
            with tempfile.TemporaryDirectory() as tmpdirname:
                self.download_model_from_huggingface(tmpdirname)
                store.upload(tmpdirname, base_model)
        
        current.card.extend(self.config_report())
        self.next(self.finetune)

    def _download_dataset_from_s3(self, tempfile_path):
        import shutil
        with S3() as s3:
            s3_resp = s3.get(self.s3_dataset_path)
            shutil.move(s3_resp.path, tempfile_path)

    @gpu_profile(interval=1)
    @kubernetes(image=HF_IMAGE, gpu=N_GPU, cpu=16, memory=72000)
    @card
    @step
    def finetune(self):
        # Load the object helping retrieve the HF Model from S3
        hf_model_store = ModelStore(
            model_store_root=self.hf_models_cache_root
        )
        
        # Create an object that will help save the model once it's done training
        self.trained_model_path = self.runtime_models_root
        trained_model_store = ModelStore(
            model_store_root=self.trained_model_path
        )
        
        # Load the dataset from S3 if it is present. 
        dataset_temp_file = None
        if self.s3_dataset_path is not None:
            print(f"Using dataset path {self.s3_dataset_path} provided by the user")
            dataset_temp_file = tempfile.NamedTemporaryFile(suffix=".json")
            self._download_dataset_from_s3(dataset_temp_file.name)

        # If there was no model that was cached on S3 then raise and Exception.
        base_model = self.config.model.base_model
        if not hf_model_store.already_exists(base_model):
            raise ValueError(f"Model {base_model} not found in the model store. This shouldn't happen.")

        # Download the model to a temporary directory and run the training
        with tempfile.TemporaryDirectory() as tmpdirname:
            hf_model_store.download(base_model, tmpdirname)
            # Main function that will run the training Loop
            self.run(
                base_model_path=tmpdirname,
                dataset_path=dataset_temp_file.name if dataset_temp_file is not None else None,
            )
            # Once training is complete. 
            model_save_dir = self.config.model.model_save_directory
            trained_model_store.upload(model_save_dir, base_model)
        
        current.card.extend(self.config_report())
        self.next(self.end)

    @step
    def end(self):
        print("Completed!")        

if __name__ == "__main__":
    LlamaInstructionTuning()
