# FlowIS: An explicitly probabilistic accelerated testing environment for autonomous driving

### 1.Project Overview
FlowIS is a probability-driven scene-behavior joint accelerated evaluation method designed to address the core challenge of autonomous vehicle (AV) safety evaluation: the inefficiency of testing rare safety-critical events (e.g., collisions, near-misses) in naturalistic driving environments (NDEs).

FlowIS solves this by:Jointly modeling high-risk initial scenes and adversarial driving behaviors (instead of isolating them, as in existing methods).
Using Normalizing Flows (NF) to shift the joint distribution of scenes and behaviors toward high-risk regions, increasing exposure to safety-critical events.
Applying Importance Sampling (IS) to reweight probabilities, ensuring unbiased evaluation results aligned with real-world NDEs.

Base Simulator: Highway-Env.
Custom Modifications: Enabled flexible probability control of initial scenes and background vehicle behaviors.

### 2.Methodology
<img src="Figure/Figure 1.svg" alt="项目演示效果">


### 3.Pre-development configuration requirements

1. The original Highway-Env environment, refer to [Highway-env](https://github.com/eleurent/highway-env)
2. The test object algorithm that can be accessed; in this project, the rl-agents is adopted, refer to [rl-agents](https://github.com/eleurent/rl-agents)
3. A runtime environment suitable for PyTorch

### 3.1 Editable install (recommended)

Use an isolated environment, then install this repository in editable mode:

```bash
conda create -n flowis python=3.11 -y
conda activate flowis
pip install -e .
```

If you also need NF/MAF training code, install torch extras:

```bash
pip install -e ".[torch]"
```

If you plan to run `Experiments/HighD_env.ipynb` (rl-agents + notebook tools):

```bash
pip install -e ".[torch,experiments]"
```

You can run a preflight import check before opening the notebook:

```bash
python scripts/check_highd_setup.py
```

### 3.2 Windows UTF-8 and Chinese path policy (important)

Root cause of `????` text in notebook/comments on Windows:

- PowerShell/terminal may run with a non-UTF-8 code page (commonly GBK/936).
- When code/notebook content with Chinese is piped through tools using mismatched encoding,
  unsupported characters are replaced with `?` before writing files.

Rules to avoid this permanently:

1. Always read/write source files with explicit UTF-8.
2. For notebook JSON edits, use `encoding='utf-8'` and `ensure_ascii=False`.
3. Avoid passing Chinese literals through shell pipelines unless UTF-8 is forced.
4. Prefer one of:
   - set UTF-8 environment before scripting (`PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`);
   - or use Unicode escapes (`\\uXXXX`) in generated strings.
5. Before committing notebook/path edits, scan for accidental `????`.

Recommended Windows session setup:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
```

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
Note that you no longer need to copy files into `site-packages/highway_env` if you use editable install.

### Citation
If you use FlowIS in your research, please cite the original paper:
