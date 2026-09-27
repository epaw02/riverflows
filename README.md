# RiverFlows 

A research project exploring river flow estimation from imagery using deep learning. It compares three architectures: **VGG19**, **Vision Transformer (ViT)**, and **RSMamba**, each adapted for regression.

The workflow includes data preparation, model training, performance evaluation, and interpretation of predictions.

## Models

- **VGG19** — a convolutional neural network used as a baseline.
- **Vision Transformer (ViT)** — an attention-based model that processes images as sequences of patches.
- **RSMamba** — a model based on the Mamba architecture, adapted for river flow estimation.

## Data and preprocessing

Images are stored in NumPy batches and linked to flow measurements through a CSV file. Preprocessing includes selecting three image channels, percentile clipping, scaling, and normalization.

Flow values are transformed using `log1p` during training. Predictions are converted back to the original scale using `expm1`.

## Training and evaluation

Models are trained using mean squared error (MSE), with validation monitoring, learning rate scheduling, and early stopping.

Evaluation includes MAE, R², and Nash–Sutcliffe efficiency (NSE), alongside plots comparing predicted and observed flow values.

## Interpretability

**Occlusion sensitivity** is used to investigate which image regions influence predictions. By masking image patches and measuring changes in model output, the project generates sensitivity heatmaps and overlays.

## Repository structure

| Path | Description |
| --- | --- |
| `datacombine.ipynb` | Data preparation and merging |
| `vgg19.ipynb` | VGG19 training and evaluation |
| `transformer.ipynb` | Vision Transformer training and evaluation |
| `rsmamba.ipynb` | RSMamba training and evaluation |
| `df_final.csv` | Combined tabular dataset |
| `csvki/` | Source CSV files |
| `images/` | Image batches in NumPy format |
| `models/` | Saved model weights |
| `charts/` | Training curves and result visualizations |
| `occlusion/` | Additional interpretability analyses and predictions |

## Getting started

1. Set up a Python environment with Jupyter, PyTorch, torchvision, NumPy, pandas, Matplotlib, scikit-learn, and timm.
2. For RSMamba, additionally configure its source code, mmengine, mmpretrain, and the required checkpoint.
3. Make the dataset and image batches available locally.
4. Update data, checkpoint, and output paths in the selected notebook.
5. Run the notebook cells sequentially.
