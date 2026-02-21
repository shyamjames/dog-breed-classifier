# Dog Image Classification

Fine-grained image classification for 120 dog breeds using transfer learning with PyTorch (MobileNetV2 backbone) and a Streamlit web app for inference.

## Features
- Transfer learning on Stanford Dogs (120 classes)
- Two-phase training (head training, then full-model fine-tuning)
- GPU support via CUDA when available (`cuda:0` fallback to CPU)
- Streamlit interface for interactive predictions

## Project Structure
- `app.py` - Streamlit app for model inference
- `src/model.py` - Model definition and classifier head setup
- `src/train.py` - Training and validation loops
- `src/prepare_data.py` - Dataset extraction/splitting utilities
- `check_gpu.py` - Quick CUDA availability check
- `requirements.txt` - Python dependencies

## Setup
1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. (Recommended) Install CUDA-enabled PyTorch wheel if you want GPU training:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## Data Preparation
Prepare the dataset with:

```powershell
python src/prepare_data.py
```

Expected layout:
- `data/raw/train/...`
- `data/raw/val/...`

## Training
Run training:

```powershell
python src/train.py
```

The script automatically selects device:
- Uses GPU when `torch.cuda.is_available()` is `True`
- Otherwise uses CPU

Best checkpoint is saved to:
- `models/best_model.pth`

## Run the App
Start the Streamlit UI:

```powershell
streamlit run app.py
```

## Notes
- If `torch.__version__` ends with `+cpu`, your environment is CPU-only.
- Verify GPU availability with:

```powershell
python check_gpu.py
```
