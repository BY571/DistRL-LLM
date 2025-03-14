> **_NOTE:_**  This repository is a work in progress. Changes and updates may occur as the project evolves.


# Distributed RL for LLM Fine-tuning

This repo contains code for distributed RL for fast and memory efficient LLM fine-tuning using [Unsloth](https://github.com/unslothai/unsloth), [Ray](https://github.com/ray-project/ray), and [vLLM](https://github.com/vllm-project/vllm).
Currently, it is configured for a simple math task, but can be extended to other tasks. 

## 🚀 Features
- Multi-GPU Training: Efficiently utilizes multiple GPUs for distributed RL-based fine-tuning.
- Memory-Efficient Fine-Tuning: Uses [Unsloth](https://github.com/unslothai/unsloth) for reduced memory footprint.
- Fast Inference: Leverages [vLLM](https://github.com/vllm-project/vllm) for high-throughput generation.
- Scalable Distributed RL: Implements [Ray](https://github.com/ray-project/ray) to orchestrate multi-GPU workloads.
- Flexible Resource Allocation: Supports customizable actor-to-learner GPU ratios (e.g., 3:1, 2:2, 1:3), allowing you to optimize resource utilization based on your specific workload characteristics and hardware configuration.

### 🏗️ Architecture Overview
This repository employs [Ray](https://github.com/ray-project/ray) for distributed computing with the following components:
- Actors: Generate candidate responses in parallel across multiple GPUs.
- Learner(s): Updates the policy based on rewards (can also participate in generation to avoid idleness).
    - **Multi-Learner Support**: Distributes gradient computation across multiple GPUs, with automatic gradient synchronization and averaging for faster and more stable training.
- Trainer: Orchestrates the entire training pipeline.
The system leverages [vLLM](https://github.com/vllm-project/vllm) for fast inference and [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient fine-tuning, making it possible to train large language models with reinforcement learning on limited hardware. This architecture enables efficient multi-GPU utilization, significantly accelerating the training process.

<details>
<summary><h1>Setup</h1></summary>

Create a new conda environment and install the dependencies:
```bash
conda create --name distrl \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate distrl
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

</details>



# Usage
After setting up the environment, you can run the distributed training with:
```bash
python train_distributed.py --run_name your_run_name --number_of_actors 2 --number_of_learners 1 --learner pg
```
You can customize various parameters:
- `--model`: The model to use (default: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
- `--dataset`: The dataset to use (default: "HuggingFaceH4/MATH-500") If you use a different dataset make sure to adapt the reward function and make sure the dataset is formatted correctly.
- `--number_of_actors`: Number of actor GPUs (default: 2)
- `--number_of_learners`: Number of learner GPUs (default: 1)
- `--batch_size`: Total batch size for all actors and learners that is later split into chunks for each actor and learner
- `--learner`: Learning algorithm to use, either "pg" (Policy Gradient) or "grpo" (Group Relative Policy Optimization)
- `--learner_chunk_size`: Sub batch size from the inital batch size for each learner to generate.
- `--topk`: Number of top-k candidates to consider for training. As we can sample thousands of completions in parallel learning is not the bottleneck with topk we can subselect the best candidates to train on.

For further parameters see `python train_distributed.py --help`.


# Learner ALgorithms
Currently, we have implemented a vanilla Policy Gradient algorithm and GRPO.
| Algorithm | Description | Status |
|-----------|-------------|--------|
| Policy Gradient | Vanilla policy gradient implementation | ✅ Implemented |
| GRPO | Group Relative Policy Optimization ([1](https://arxiv.org/abs/2402.03300), [2](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)) | ✅ Implemented |



<details>
<summary><h1>Performance Tests</h1></summary>

Policy gradient training (~2hours) Model: Qwen2.5-7B-Instruct-bnb-4bit, Dataset: MATH-500

![Performance Tests PG](./media/initial_pg_test.png)


GRPO training (~2hours) Model: Qwen2.5-7B-Instruct-bnb-4bit, Dataset: MATH-500

![Performance Tests GRPO](./media/init_grpo_test.png)

</details>

## TODO:
- ~~Usually the bottleneck for RL with LLMs is the online data generation. However, with ray and vllm this is not a problem. Instead the bottleneck is now the learning as we can sample thousands of completions in parallel. *Can we parallelize the learning process?*~~ ✅ Solved with multi-learner support
- Add more learner algorithms
- Make option to use w & w/o vllm
- Training becomes unstable with longer training, try to fix
