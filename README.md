# Edge AI Traffic Sign Classifier

This project contains a PyTorch Lightning implementation for an Edge AI traffic sign model designed for resource-constrained hardware like the ESP32-S3. The model is trained and automatically exported to ONNX format.

## Setup and Installation

This project manages its dependencies and virtual environment using [uv](https://github.com/astral-sh/uv).

Since you have already created the `.venv` using `uv`, you can activate the environment and run the code with the following steps.

### 1. Activate the Virtual Environment

On **Windows** (using Command Prompt or PowerShell):
```powershell
.venv\Scripts\activate
```

*(On macOS/Linux, it would be `source .venv/bin/activate`)*

### 2. Install Dependencies (If not already installed)

If you haven't yet synchronized your dependencies, install them into the active virtual environment:
```powershell
# Using uv to sync or install packages
uv pip install torch torchvision pytorch-lightning onnx
```
*(Depending on whether you have a `requirements.txt` or `pyproject.toml`, you could also run `uv pip install -r requirements.txt`)*

## Running the Code

With the virtual environment active, you can start the training and export process by running the main entry point script:

```powershell
python main.py
```

Alternatively, `uv` allows you to run scripts directly without explicitly activating the environment:
```powershell
uv run py/python main.py
```

### What `main.py` Does:
1.  **Loads Data:** Reads the traffic sign dataset from the `data/` directory.
2.  **Builds the Model:** Instantiates a lightweight Separable CNN + Global Average Pooling architecture optimized for ESP32-S3 (<200k parameters).
3.  **Trains:** Runs PyTorch Lightning training with early stopping and automatic checkpointing to `models/checkpoints/`.
4.  **Evaluates:** Performs inference on the test set if available.
5.  **Exports:** Automatically converts the best model checkpoint into an ONNX payload at `models/exports/edge_ai_traffic_sign.onnx`.
6.  **Reports:** Prints the final model footprint (parameter count and ONNX size).

## Project Structure

```text
pioneer/
├── src/                    # Core source code
│   ├── models/             # Model architectures (traffic_sign_cnn.py)
│   ├── data/               # Data loading and preprocessing (dataset.py)
│   └── training/           # Training modules and logic (lightning_module.py)
├── scripts/                # Executable scripts
│   ├── training/           # Training scripts (PyTorch and TensorFlow)
│   ├── evaluation/         # Evaluation, testing, and TFLite benchmarks
│   ├── export/             # Model export and conversion (ONNX, TFLite)
│   └── utils/              # Utility scripts (model_info.py)
├── models/                 # Storage for model checkpoints and exports
│   ├── checkpoints/        # PyTorch .ckpt files
│   ├── checkpoints_tf/     # TensorFlow .keras files
│   └── exports/            # Final ONNX and TFLite models
├── data/                   # Dataset directory (train/test/sample_submission.csv)
├── results/                # Output directory for submission CSVs
├── assets/                 # Project assets (images, documentation)
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

## Advanced Usage (Scripts)

You can run specialized scripts for individual tasks:

- **Training:**
  - `python scripts/training/train_pytorch.py`: Standalone PyTorch training.
  - `python scripts/training/train_tensorflow.py`: LeNet-5 training in TensorFlow.
- **Evaluation:**
  - `python scripts/evaluation/val_pytorch.py`: Evaluate PyTorch checkpoint on validation set.
  - `python scripts/evaluation/test_pytorch.py`: Generate Kaggle submission from PyTorch model.
  - `python scripts/evaluation/test_tflite_int8.py`: Benchmark INT8 quantized TFLite inference.
- **Export:**
  - `python scripts/export/onnx_to_tflite.py`: Multi-stage conversion (PT -> ONNX -> TF -> TFLite).
  - `python scripts/export/keras_to_tflite.py`: Convert Keras model to TFLite (Float32 & INT8).
- **Utility:**
  - `python scripts/utils/model_info.py`: Detailed architecture and parameter analysis.