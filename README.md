# FlowIS:An explicitly probabilistic accelerated testing environment for autonomous driving

ProjectName and Description

### 1.Project Overview
FlowIS is a probability-driven scene-behavior joint accelerated evaluation method designed to address the core challenge of autonomous vehicle (AV) safety evaluation: the inefficiency of testing rare safety-critical events (e.g., collisions, near-misses) in naturalistic driving environments (NDEs).

FlowIS solves this by:Jointly modeling high-risk initial scenes and adversarial driving behaviors (instead of isolating them, as in existing methods).
Using Normalizing Flows (NF) to shift the joint distribution of scenes and behaviors toward high-risk regions, increasing exposure to safety-critical events.
Applying Importance Sampling (IS) to reweight probabilities, ensuring unbiased evaluation results aligned with real-world NDEs.

Base Simulator: Highway-Env.
Custom Modifications: Enabled flexible probability control of initial scenes and background vehicle behaviors.

### 2.Methodology



### 3.Pre-development configuration requirements

1. The original Highway-Env environment, refer to XXX
2. The test object algorithm that can be accessed; in this project, the rl-agent is adopted, refer to XXX
3. A runtime environment suitable for PyTorch

### 4.File Directory Description
eg:
```
FlowIS/
├── LICENSE.txt
├── /Experiments/
│   ├── core_code/
│   │   ├── NormalizingFlow.py
│   │   ├── behavior.py
│   │   ├── highway_highD_env.py
│   │   ├── intersection_inD_env.py
│   │   └── utils_FlowIS.py
│   ├── HighD_env.ipynb
│   └── InD_env.ipynb
└── /FlowIS/
│   ├── Normalizing_Flow/
│   ├── envs/
│   ├── road/
│   ├── vehicle/
│   ├── __init__.py
│   ├── interval.py
│   └── utils.py
└── README.md
```
Note that when reproducing the experiments, the corresponding files in FlowIS should be replaced into your Highway-Env environment to achieve a complete configuration.

### Citation
If you use FlowIS in your research, please cite the original paper:
