import glob
import os

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

def _format_to_instruction_tune(data_arr):
    needed_keys = ["instruction", "output" ]
    # If the input is in alpaca format then let it be
    if all([dict_contains_all_keys(d, needed_keys) for d in data_arr]):
        print("Data contains all necessary keys")
        return data_arr
    
    # If the input is not in Alpaca format then convert it 
    final_arr = []
    key_map = {
        "response": "output",
        "instruction": "instruction",
        "context": "input"
    }
    for d in data_arr:
        data_dict = {}
        for k in key_map:
            if k not in d:
                data_dict[key_map[k]] = None
            else:
                data_dict[key_map[k]] = d[k]
        final_arr.append(data_dict)
    return final_arr
        

def transform_data_to_instruction_tune(data_path):
    json_files = find_json_files_in_directory(data_path)
    all_data = []
    for data_arr in load_json_files(json_files):
        all_data.extend(
            _format_to_instruction_tune(data_arr)
        )
    return all_data
