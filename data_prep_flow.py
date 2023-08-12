from metaflow import FlowSpec, step, project, kubernetes, Parameter, S3, card
# from mixins import HF_IMAGE
from custom_decorators import pip
import os
import glob


def find_json_files_in_directory(directory):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))

    all_files = json_files + jsonl_files
    return all_files

def load_json_files(file_paths):
    
    def _load_json_file(path):
        import json
        with open(path, "r") as f:
            return json.load(f)
    
    def _load_jsonl_file(path):
        import json
        with open(path, "r") as f:
            return [json.loads(line) for line in f.readlines()]
    
    for f in file_paths:
        if f.endswith(".json"):
            yield _load_json_file(f)
        elif f.endswith(".jsonl"):
            yield _load_jsonl_file(f)

def dict_contains_all_keys(d, keys):
    return all([k in d for k in keys])

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
        is_flag=True
    )

    def _upload_dataset(self, data_path):
        json_files = find_json_files_in_directory(data_path)
        with S3(run=self) as s3:
            s3_resp = s3.put_files([
                (
                os.path.join("dataset",os.path.basename(f)),f
                ) for f in json_files
            ], )
            return s3_resp
    
    def _format_to_instruction_tune(self, data_arr):
        needed_keys = ["instruction", "output"]
        # If the input is in alpaca format then let it be
        if all([dict_contains_all_keys(d, needed_keys) for d in data_arr]):
            print("Data contains all necessary keys")
            return data_arr
        
        # If the input is not in Alpaca format then convert it 
        final_arr = []
        key_map = {
            "response": "output",
            "instruction": "instruction"
        }
        for d in data_arr:
            data_dict = {}
            for k in key_map:
                if k not in d:
                    raise Exception (f"Key {k} not found in the data")
                data_dict[key_map[k]] = d[k]
            final_arr.append(data_dict)
        return final_arr
            

    def _transform_data(self, data_path):
        json_files = find_json_files_in_directory(data_path)
        all_data = []
        for data_arr in load_json_files(json_files):
            all_data.extend(
                self._format_to_instruction_tune(data_arr)
            )
        return all_data
                
    @pip(libraries={"huggingface-hub":"0.16.4"})
    @card
    @step
    def start(self):
        assert len(self.hf_dataset_path.split("/")) > 1
        from huggingface_hub import snapshot_download
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = snapshot_download(repo_id=self.hf_dataset_path, repo_type="dataset", local_dir=tmpdir, local_dir_use_symlinks=False)
            all_data_dict = self._transform_data(data_path)
        
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
        
        if self.raise_event:
            if len(self.remote_dataset_path) > 0:
                ArgoEvent(
                    name=f"{current.project_name}.dataprep",
                ).publish(
                    payload={
                        # remote_dataset_path should only have one file since we join all the files together.
                        "s3_dataset_path": self.remote_dataset_path[0],
                    }
                )
            else:
                print("No dataset path found. No event was raised.")
        print("Completed!")

if __name__ == "__main__":
    DataPrepFlow()