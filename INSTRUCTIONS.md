# SRGAN Super-Resolution GAN - Usage Instructions

## Overview
This repository contains a Super-Resolution Generative Adversarial Network (SRGAN) implementation for upscaling low-resolution images to high-resolution using deep learning.

## Prerequisites

### Environment Setup
1. **Conda Environment**: Use the `humam_dev` environment (already configured)
2. **Dependencies**: PyTorch, torchvision, numpy, pillow, scikit-image (already installed)

### Dataset Requirements
- **Training**: Paired low-resolution (LR) and high-resolution (HR) images
- **Testing**: Low-resolution images only
- **Supported formats**: PNG, JPG, JPEG, BMP, TIF, TIFF, WEBP

## Quick Start

### 1. Activate Environment
```bash
eval "$(conda shell.zsh hook)" && conda activate humam_dev
```

### 2. Test with Pretrained Model
```bash
python3 main.py --mode test_only --LR_path "path/to/your/LR/images" --generator_path "pretrained_models/SRGAN.pt"
```

### 3. Train New Model
```bash
python3 main.py --LR_path "path/to/LR/images" --GT_path "path/to/HR/images"
```

## Detailed Usage

### Test-Only Mode (Inference)
Generate super-resolution images using a pretrained model:

```bash
python3 main.py \
    --mode test_only \
    --LR_path "test_data" \
    --generator_path "pretrained_models/SRGAN.pt"
```

**Output**: Super-resolution images saved to `result/` folder

### Training Mode
Train a new SRGAN model from scratch:

```bash
python3 main.py \
    --LR_path "Data/train_LR" \
    --GT_path "Data/train_HR" \
    --batch_size 16 \
    --pre_train_epoch 8000 \
    --fine_train_epoch 4000
```

**Output**: Trained models saved to `model/` folder

### Testing Mode (With Ground Truth)
Evaluate model performance with PSNR metrics:

```bash
python3 main.py \
    --mode test \
    --LR_path "Data/test_LR" \
    --GT_path "Data/test_HR" \
    --generator_path "model/SRGAN_gene_4000.pt"
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--LR_path` | str | `../DIV2K/DIV2K_train_LR_bicubic/X4` | Path to low-resolution images |
| `--GT_path` | str | `../DIV2K/DIV2K_train_HR/` | Path to ground truth high-resolution images |
| `--mode` | str | `train` | Mode: `train`, `test`, or `test_only` |
| `--generator_path` | str | None | Path to pretrained generator model |
| `--batch_size` | int | 16 | Training batch size |
| `--pre_train_epoch` | int | 8000 | L2 loss pre-training epochs |
| `--fine_train_epoch` | int | 4000 | GAN fine-tuning epochs |
| `--scale` | int | 4 | Upscaling factor |
| `--patch_size` | int | 24 | Training patch size |
| `--res_num` | int | 16 | Number of residual blocks |
| `--num_workers` | int | 0 | DataLoader workers |
| `--in_memory` | bool | True | Load all images into memory |

## Dataset Structure

### For Training
```
Data/
├── train_LR/          # Low-resolution images
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── train_HR/          # High-resolution images (4x larger)
    ├── image1.png
    ├── image2.png
    └── ...
```

### For Testing
```
test_data/             # Low-resolution images for inference
├── test1.png
├── test2.png
└── ...
```

## Training Process

The training consists of two phases:

### Phase 1: Pre-training (L2 Loss)
- Trains generator with MSE loss only
- Saves checkpoints every 800 epochs
- Duration: `--pre_train_epoch` iterations

### Phase 2: Fine-tuning (Adversarial Loss)
- Trains both generator and discriminator
- Uses perceptual loss + adversarial loss
- Saves models every 500 epochs
- Duration: `--fine_train_epoch` iterations

## Output Files

### Training Outputs
- `model/pre_trained_model_XXXX.pt` - Pre-training checkpoints
- `model/SRGAN_gene_XXXX.pt` - Final generator model
- `model/SRGAN_discrim_XXXX.pt` - Final discriminator model

### Testing Outputs
- `result/res_XXXX.png` - Super-resolution images
- `result.txt` - PSNR metrics (test mode only)

## Troubleshooting

### Common Issues

1. **CUDA Error**: Models are automatically mapped to CPU
2. **No Images Found**: Ensure image files are in supported formats
3. **Memory Error**: Reduce `--batch_size` or set `--in_memory False`
4. **Missing Dependencies**: Activate `humam_dev` environment

### Performance Tips

- **CPU Training**: Use smaller batch sizes (8-16)
- **Memory**: Set `--in_memory False` for large datasets
- **Speed**: Increase `--num_workers` if you have multiple CPU cores

## Examples

### Quick Test
```bash
# Test with sample images
python3 main.py --mode test_only --LR_path "test_data" --generator_path "pretrained_models/SRGAN.pt"
```

### Custom Training
```bash
# Train with custom parameters
python3 main.py \
    --LR_path "my_data/LR" \
    --GT_path "my_data/HR" \
    --batch_size 8 \
    --pre_train_epoch 4000 \
    --fine_train_epoch 2000 \
    --scale 4
```

### Resume Training
```bash
# Continue training from checkpoint
python3 main.py \
    --LR_path "Data/train_LR" \
    --GT_path "Data/train_HR" \
    --fine_tuning True \
    --generator_path "model/pre_trained_model_8000.pt"
```

## Notes

- Training can take several hours depending on dataset size and hardware
- The pretrained model (`pretrained_models/SRGAN.pt`) is ready for immediate use
- All images are automatically normalized to [-1, 1] range
- The model outputs 4x upscaled images by default
