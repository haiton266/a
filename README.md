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
1. **Loads Data:** Loads the traffic sign images from the local `kaggle_testing/` dataset directory.
2. **Builds the Model:** Instantiates a lightweight Separable CNN + Global Average Pooling model optimized for edge devices (<200k parameters).
3. **Trains:** Runs PyTorch Lightning training with early stopping and checkpointing. 
4. **Evaluates:** Runs inference on the test set.
5. **Exports:** Converts the best performing model checkpoint into an ONNX payload (`edge_ai_traffic_sign.onnx`).
6. **Reports:** Prints an edge deployment footprint summary calculating model payload constraints.
