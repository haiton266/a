"""
val.py — Quick Validation Benchmark (PyTorch)

What is this?
------------
A standalone script to evaluate the performance of a PyTorch checkpoint 
on the validation set. Useful for quick checks without running a full 
test or training cycle.

Workflow:
  1. Load validation dataloader
  2. Load best_edge_model.ckpt
  3. Run trainer.validate()
  4. Print validation loss and accuracy
"""

import os
import sys

# Path adjustment for src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pytorch_lightning as pl
from src.training.lightning_module import TrafficSignLightningModel
from src.data.dataset import get_dataloaders

NUM_CLASSES = 10
BATCH_SIZE = 32

def val():
    print("Loading validation dataset from data/...")
    _, val_loader, _ = get_dataloaders(base_dir="data", batch_size=BATCH_SIZE, num_classes=NUM_CLASSES)
    
    checkpoint_path = 'models/checkpoints/best_edge_model.ckpt'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found. Please train the model first.")
        return
        
    print(f"Loading checkpoint from {checkpoint_path}")
    model = TrafficSignLightningModel.load_from_checkpoint(checkpoint_path)
    
    trainer = pl.Trainer(logger=False, enable_progress_bar=True)
    
    print("\nStarting Validation...")
    results = trainer.validate(model, val_loader)
    print(f"\nValidation Results: {results}")

if __name__ == "__main__":
    val()
