### For context, read this blog article: [Tuning Large Language Models to Follow Instructions on Metaflow](https://outerbounds.com/blog/llm-tuning-metaflow/)

# Tuning Large Language Models to Follow Instructions on Metaflow
This repository offers a framework for instruction tuning using HuggingFace and Metaflow. It includes a general workflow template you can readily adapt to many HuggingFace models while Metaflow makes it easy and cost-effective to [scale](https://docs.metaflow.org/scaling/introduction) as needed. If you wish to urn the code on your own Metaflow deployment, you can choose from these [deployment guides](https://outerbounds.com/engineering/welcome/). Please reach us on [Slack](http://slack.outerbounds.co/) if you have questions or need help getting set up!

![the gif shows instruction tuning of a large language model](img/tuning-llm.gif)

## Setup

### GPU infrastructure
You will want to run the code on GPUs. In our experiments, we used [CoreWeave](https://www.coreweave.com/) to access hardware, and found the following configurations to work well with price points ranging from $5 to $10 per hour. Configurations we had success with are listed below, ordered by most expensive processors first:
- 2 NVIDIA A100 80GB for PCIe
- 2 NVIDIA A100 40GB for PCIe
- 2 NVIDIA RTX A6000 (48GB memory)

If desired, you can run the workflows with GPUs configured in your own Metaflow deployment. If you don't have infrastructure setup, you can set it up with one of our [CloudFormation](https://github.com/outerbounds/metaflow-tools/blob/master/aws/cloudformation/metaflow-cfn-template.yml) or [Terraform](https://github.com/outerbounds/terraform-aws-metaflow) templates. To deploy the GPU infrastructure on AWS, change the `ComputeEnvInstanceTypes` field in CloudFormation template or AWS console UI to an instance type supporting the needed GPU types. More detailed instructions on setting up Metaflow infrastructure can be found [here](https://outerbounds.com/engineering/welcome/).

### Python environment
You can install dependencies by using the `requirements.txt` file:
```
pip install -r requirements.txt
```

Note that if you are running in local mode, you won't need to use the custom `@pip` decorator that is in the `flow.py` file by default if you do the above step, since the requirements only need to be installed once in the local mode case. 

### Pushing checkpoints to HuggingFace
You can optionally push checkpoints on HuggingFace when you run the flow. To do so, the code requires a [HuggingFace token](https://huggingface.co/docs/hub/security-tokens) and a HuggingFace organization to be set as environment variables, `HUGGINGFACE_TOKEN` and `HF_ORGANIZATION` respectively, in the command line environment you run Metaflow commands from.

```
export HUGGINGFACE_TOKEN=<INSERT YOUR TOKEN>
export HF_ORGANIZATION=<INSERT YOUR ORGANIZATION>
```

Note that this is not required to run the code and pushing checkpoints is disabled by default.

## Run the code

### Fine-tuning
The datasets and models are already on the HuggingFace Hub, and the code is configured to pull them. You can run the workflow in the `flow.py` file with this Metaflow command:

#### Custom Dataset

Thanks to [tloen/alpaca-lora repository](https://github.com/tloen/alpaca-lora) we support custom JSON datasets. 
Format should be as follows

```
[ 
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1. Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    },
    {
        "instruction": "What are the three primary colors?",
        "input": "",
        "output": "The three primary colors are red, blue, and yellow."
    },
] 
```

#### Large Language Models

By default, code runs using LLaMA 7B as a base model, but it should be easy
to use any other model from HuggingFace. Simply change the file to `model.base_model` attribute in the configuration file ([experiment_config.yaml](./experiment_config.yaml)). 

#### Local run
```
python flow.py run --config-file experiment_config.yaml
```

#### Running on kubernetes
```
python remote_flow.py run --config-file experiment_config.yaml
```

#### Workflow description
Under the hood, the workflow calls the `tuner.py` file using a call to `torchrun` in the `tune` step, which is originally sourced from the [tloen/alpaca-lora repository](https://github.com/tloen/alpaca-lora). This template can be readily adapted for many types of HuggingFace model. Feel free to fork and customize this repository to your needs - for example to expose more of the parameters for training, to try new modeling approaches, custom data, and to add/modify steps in the workflow. 

### Evaluation
For the evaluation we use the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) package. Although not enabled by default since it requires extra steps, we do recommend pushing the models on the HuggingFace Hub if you expect to revisit the model training in the future or wish to do more customized evaluation and deployment processes with the resulting model. 

## Support
Please reach us on [Slack](http://slack.outerbounds.co/) if you need help or have questions about the workflow.