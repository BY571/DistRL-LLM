> **_NOTE:_**  This repository is a work in progress. Changes and updates may occur as the project evolves.


# Distributed RL for LLM Fine-tuning

This repo contains code for distributed RL for fast and memory efficient LLM fine-tuning with unsloth, ray and vllm. 
Currently, it is configured for a simple math task, but can be extended to other tasks. 

# Setup



# Learner ALgorithms
Currently, we have implemented a vanilla Policy Gradient algorithm and GRPO.
| Algorithm | Description | Status |
|-----------|-------------|--------|
| Policy Gradient | Vanilla policy gradient implementation | ✅ Implemented |
| GRPO | Generalized Reward-Weighted Policy Optimization | ✅ Implemented |

# TODO:
- Usually the bottleneck for RL with LLMs is the online data generation. However, with ray and vllm this is not a problem. Instead the bottleneck is now the learning as we can sample thousands of completions in parallel. *Can we parallelize the learning process?*
- Add more learner algorithms
- Make option to use w & w/o vllm