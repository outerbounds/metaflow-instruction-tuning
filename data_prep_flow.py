from metaflow import FlowSpec, step, project, kubernetes, Parameter, S3, card, pypi
# from mixins import HF_IMAGE
import os
from hf_data_prep_utils import transform_data_to_instruction_tune, find_json_files_in_directory

@project(name="alpaca")
class DataPrepFlow(FlowSpec):

    hf_dataset_path = Parameter(
        "hf-dataset-path",
        help = "HuggingFace dataset path",
        default = "databricks/databricks-dolly-15k",
    )

    raise_event = Parameter(
        "raise-event",
        help = "Raise event to trigger training",
        default = False,
        type = bool,
    )
                
    @pypi(packages={"huggingface-hub":"0.16.4"})
    @card
    @step
    def start(self):
        assert len(self.hf_dataset_path.split("/")) > 1
        from huggingface_hub import snapshot_download
        import tempfile
        # Download the Dataset from huggingface and transform it to Alpaca format
        # transform all the files in the dataset to one file and then upload it to S3
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = snapshot_download(repo_id=self.hf_dataset_path, repo_type="dataset", local_dir=tmpdir, local_dir_use_symlinks=False)
            all_data_dict = transform_data_to_instruction_tune(data_path)
        
        # Once the dataset is transformed, upload it to S3. 
        with tempfile.TemporaryDirectory() as final_data_path:
            import json
            with open(os.path.join(final_data_path, f"{self.hf_dataset_path.split('/')[-1]}.json"), "w") as f:
                json.dump(all_data_dict, f)
            self.remote_dataset_path = self._upload_dataset(final_data_path)
                
        self.next(self.end)
    
    @step
    def end(self):
        from metaflow.integrations import ArgoEvent
        from metaflow import current
        
        # Once the dataset is uploaded, raise an event to trigger training
        if self.raise_event:
            if len(self.remote_dataset_path) > 0:
                key, s3_path = self.remote_dataset_path[0]
                ArgoEvent(
                    name=f"{current.project_name}.dataprep",
                ).publish(
                    payload={
                        # remote_dataset_path should only have one file since we join all the files together.
                        "dataset_path": s3_path,
                    }
                )
            else:
                print("No dataset path found. No event was raised.")
        print("Completed!")
    
    def _upload_dataset(self, data_path):
        json_files = find_json_files_in_directory(data_path)
        with S3(run=self) as s3:
            s3_resp = s3.put_files([
                (
                os.path.join("dataset",os.path.basename(f)),f
                ) for f in json_files
            ], )
            return s3_resp

if __name__ == "__main__":
    DataPrepFlow()