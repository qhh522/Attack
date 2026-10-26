
Adversarial attack experiments based on MMDetection.

**Paper**: [https://www.sciencedirect.com/science/article/pii/S0957417426032963](https://www.sciencedirect.com/science/article/pii/S0957417426032963)

## Environment Setup

```bash
# Create virtual environment
conda create -n kd2 python=3.10 -y && conda activate kd2

# PyTorch (choose CUDA 11.8 or 12.1 as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# MMDetection
pip install openmim && mim install mmdet

# Other dependencies
pip install omegaconf pillow numpy opencv-python
```

Verify: `python -c "from mmdet.apis import init_detector; print('OK')"`

## Usage

```bash
# Adversarial attack experiments 
python attack/attack.py
```

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `outer_loops` | Outer loop iterations | 
| `inner_loops` | Inner loop iterations |
| `alpha` | Attack step size | 
| `eps` | Perturbation bound |
| `gamma` | Dual-model branch weight | 

> Ensure model weights (`.pth`) are placed in the corresponding `work_dirs_*/` directories before running. Adjust dataset paths in scripts as needed.