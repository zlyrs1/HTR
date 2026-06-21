# HTR: Hierarchical Task Reasoning for Aerial Vision-and-Language Navigation

Liangyu Zhou, Rui Xue, Xiaoyan Luo

School of Electronic and Information Engineering, Beihang University  
School of Astronautics, Beihang University

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-ee4c2c.svg)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-AerialVLN-lightgrey.svg)](https://github.com/AirVLN/AirVLN)

## Abstract

Vision-and-Language Navigation (VLN) requires an embodied agent to navigate through an environment by following natural language instructions. Compared with ground-based VLN, aerial VLN is more challenging because UAV agents operate in large-scale outdoor scenes with long-horizon instructions, complex spatial relations, and continuous 3D motion.

We propose **Hierarchical Task Reasoning (HTR)** for aerial VLN. HTR uses an offline Large Language Model (LLM) parser to convert each instruction into a hierarchical task tree containing subtasks, action components, landmark components, and primitive elements. During navigation, the policy first selects the active subtask for coarse-grained contextual reasoning, then performs fine-grained action and landmark refinement. The landmark stream uses Landmark Slot Attention (LSA) to ground fine-grained landmark elements in CLIP visual patch tokens.

This repository contains the PyTorch implementation of HTR on the AerialVLN navigation environment.

## Method

HTR contains three main components:

- **Hierarchical task tree construction**: decomposes each instruction into `Subtask`, `nA`, `nL`, `Ae`, and `Le` fields. This step is performed offline before training or evaluation.
- **Coarse-grained subtask reasoning**: uses the recurrent navigation state to select the currently active subtask through a hard subtask gate.
- **Fine-grained action and landmark refinement**: refines action elements and landmark elements in two streams, where the landmark stream uses LSA for object-centric visual grounding.

In the codebase, the main implementation is in:

- `src/vlnce_src/env.py`: loads AerialVLN episodes and encodes hierarchical instruction fields with the CLIP text encoder.
- `Model/cma_policy.py`: implements the HTR policy, including subtask selection, component attention, element attention, and Landmark Slot Attention.
- `src/vlnce_src/train.py`: contains training and evaluation loops for the AirSim-based AerialVLN environment.

## Prerequisites

### Installation

HTR follows the AerialVLN environment setup. We recommend using Ubuntu, NVIDIA GPUs, CUDA, and Conda.

Create a workspace and install the repository:

```bash
mkdir HTR_ws
cd HTR_ws

git clone https://github.com/zlyrs1/HTR.git
cd HTR
```

Create and activate a conda environment:

```bash
conda create -n HTR python=3.8
conda activate HTR

pip install pip==24.0 setuptools==63.2.0
pip install -r requirements.txt
```

Install PyTorch according to your CUDA version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
```

This code is designed for the AirSim-based AerialVLN simulator. Please follow the official AerialVLN instructions to prepare the simulator, datasets, and pretrained PointNav depth encoder.

### Model, Simulator, and Datasets

Prepare the workspace in the same style as AerialVLN:

```text
HTR_ws
|-- HTR
|-- DATA
|   |-- data
|   |   |-- aerialvln
|   |   `-- aerialvln-s
|   |-- img_features
|   |-- models
|   |   |-- ddppo-models
|   |   `-- resnet50
|   `-- output
`-- ENVs
    |-- env_1
    |-- env_2
    `-- ...
```

Required external resources:

- **AerialVLN simulators**: download and place the AirSim/UE4 simulator scenes under `ENVs`.
- **AerialVLN and AerialVLN-S datasets**: place episode annotations under `DATA/data/aerialvln` and `DATA/data/aerialvln-s`.
- **Depth encoder checkpoint**: place `gibson-2plus-resnet50.pth` under `DATA/models/ddppo-models`.
- **CLIP checkpoint**: place `CLIP-ViT-B-32-laion2B-s34B-b79K.bin` under `HTR/src/vlnce_src/laion`.

The default project root is configured through `--project_prefix` in `src/common/param.py`. On the server used for our experiments, this path is `/home/code`. You can override it from the command line:

```bash
python -u ./src/vlnce_src/train.py --project_prefix /path/to/HTR_ws
```

### Hierarchical Instruction Data

HTR uses an offline Qwen-based parsing pipeline to construct the hierarchical task tree before training or evaluation. The parser reads AerialVLN-style episode JSON files and augments each episode's `instruction` field with multi-level semantic fields used by the navigation policy.

The three-stage pipeline is:

1. **Instruction standardization**: `loadInstruct00.py` uses `qwen00` to normalize raw navigation text and saves `preprocessed_text`.
2. **Component parsing**: `loadInstruct01.py` uses `qwen01` to decompose each standardized instruction into `n.A`, `b.L`, and `o.L` components, then programmatically groups them into `Subtask` and `Component`.
3. **Element extraction**: `loadInstruct02.py` extracts `nA` and `Ae` from action components, gathers landmark components as `nL`, and uses `qwen02` to extract landmark elements `Le`.

After parsing, each episode instruction should contain:

```text
instruction
|-- instruction_text
|-- preprocessed_text
|-- Subtask
|-- Component
|-- nA
|-- nL
|-- Ae
`-- Le
```

Before running the parser, create the Ollama models from the provided modelfiles:

```bash
cd Qwen
ollama create qwen00 -f qwen00.modelfile
ollama create qwen01 -f qwen01.modelfile
ollama create qwen02 -f qwen02.modelfile
```

Then set `JSON_FILE` in each `loadInstruct*.py` script to the target AerialVLN split file and run the stages sequentially:

```bash
python loadInstruct00.py
python loadInstruct01.py
python loadInstruct02.py
```

During model training and evaluation, `src/vlnce_src/env.py` encodes `Subtask`, `nA`, `nL`, `Ae`, and `Le` with the CLIP text encoder and provides them to the policy as:

```text
subtask_embedding
nA_embedding
nL_embedding
Ae_embedding
Le_embedding
```

## Repository Structure

```text
HTR
|-- Qwen/                 # Offline Qwen parser for hierarchical task trees
|-- airsim_plugin/        # AirSim simulator client and server utilities
|-- Model/                # HTR policy, encoders, recurrent modules, losses
|-- scripts/              # Example scripts for dataset download, training, evaluation
|-- src/
|   |-- common/           # Argument and path configuration
|   `-- vlnce_src/        # AerialVLN environment, training, and evaluation loops
|-- utils/                # Environment utilities, metrics, logging, vector env wrappers
|-- requirements.txt
`-- README.md
```

## Training and Evaluation

Start the AirVLN simulator server before training or evaluation:

```bash
nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1,2,3 &
```

Train the HTR policy:

```bash
python -u ./src/vlnce_src/train.py \
  --run_type train \
  --policy_type cma \
  --collect_type TF \
  --name HTR-cma \
  --batchSize 8 \
  --dagger_it 1 \
  --epochs 100 \
  --lr 0.00025 \
  --trainer_gpu_device 0
```

Evaluate a trained checkpoint:

```bash
python -u ./src/vlnce_src/train.py \
  --run_type eval \
  --policy_type cma \
  --collect_type TF \
  --name HTR-cma \
  --batchSize 1 \
  --EVAL_CKPT_PATH_DIR /path/to/checkpoint.pth \
  --EVAL_DATASET val_unseen \
  --EVAL_NUM -1
```

Training checkpoints, logs, TensorBoard files, and evaluation results are saved under:

```text
DATA/output/{experiment_name}/
```

## Configuration

Common options are defined in `src/common/param.py`.

Important arguments include:

- `--project_prefix`: workspace root containing `HTR`, `DATA`, and `ENVs`.
- `--policy_type cma`: enables the HTR policy implementation in `Model/cma_policy.py`.
- `--collect_type TF`: trains from teacher-forcing data.
- `--collect_type dagger`: uses DAgger-style data collection.
- `--EVAL_CKPT_PATH_DIR`: checkpoint path for evaluation.
- `--EVAL_DATASET`: evaluation split, such as `val_seen` or `val_unseen`.

## Acknowledgements

This project builds on the AerialVLN simulator and navigation environment. We thank the AerialVLN authors for releasing the dataset, simulator, and baseline code.

## Contact

For questions about the paper or code, please contact:

```text
zhoulyvln@buaa.edu.cn
```
