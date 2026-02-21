# Stanford Dogs Classification

A fine-grained image classifier for 120 dog breeds using a pre-trained **MobileNetV2** model via transfer learning in PyTorch, deployed intelligently as an interactive Streamlit application.

## 🚀 Getting Started

**Prerequisites:** You need Python (recommended 3.10-3.13 if you wish to use your RTX 3050 GPU immediately via PyPI CUDA wheels) and the generated virtual environment. 

1. **Activate the virtual environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Train the Model**:
   *Note: `src/train.py` is currently set to run a fast integration test (2 batches per epoch).*
   *To run the full training process, simply open `src/train.py` and remove the `if bi >= 2: break` limit in the `train_model` function.*
   ```powershell
   python src/train.py
   ```

3. **Run the Streamlit App**:
   The app will use the saved `models/best_model.pth`.
   ```powershell
   streamlit run app.py
   ```

## 🧠 Model Architecture & Strategy
- **Base**: `MobileNetV2` trained on ImageNet.
- **Top**: Custom Dropout (0.4) + Linear(1280, 120) classification head to reduce overfitting.
- **Phase 1 Training**: Adam optimizer on the custom head only (Frozen base).
- **Phase 2 Training**: Unfrozen base structure fine-tuning with a smaller learning rate (`1e-5`).

## 📁 Repository Structure
- `app.py`: The Main Streamlit dashboard.
- `src/model.py`: PyTorch Module architecture.
- `src/train.py`: The main training and fine-tuning loops.
- `src/prepare_data.py`: Preprocessed the `.tar` dataset into `data/train`, `data/val`, `data/test`.
- `data/`: Extracted Images dataset, split ready.
- `models/`: Stores the best `pth` checkpoints.
