# Sample Flow For running on Corweave instances. 
from metaflow import FlowSpec, step, Parameter, resources, environment
from mixins import HuggingFaceLora, N_GPU, visible_devices
from custom_decorators import pip, gpu_profile
import os

class LlamaInstructionTuning(FlowSpec, HuggingFaceLora):

    push_checkpoints = Parameter("push", help="push checkpoints on huggingface", default=False, type=bool)

    @step
    def start(self):
        self.next(self.finetune)

    @environment(vars={
        "CUDA_VISIBLE_DEVICES": visible_devices, 
        "WORLD_SIZE": N_GPU,
        "HUGGINGFACE_TOKEN": os.environ["HUGGINGFACE_TOKEN"],
        "HF_ORGANIZATION": os.environ["HF_ORGANIZATION"]
    })
    @gpu_profile(interval=1)
    @pip(file="requirements.txt")
    @resources(gpu=N_GPU, cpu=16, memory=128000) # tested with A100 and A6000 GPU.
    @step
    def finetune(self):
        self.run()
        if self.push_checkpoints:
            self.upload_to_huggingface()
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    LlamaInstructionTuning()

# LOCAL: python flow.py run
# REMOTE: python flow.py --package-suffixes=.txt,.json run --with batch