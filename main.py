import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from module import TrafficSignLightningModel
from data import get_dataloaders
from export import export_to_onnx

INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 50

def main():
    print("Loading dataset from kaggle_testing/...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES)
    
    # --- Build Lightning Model ---
    model = TrafficSignLightningModel(num_classes=NUM_CLASSES)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=> Total Model Parameters: {param_count:,} (Limit: 200,000)")
    if param_count > 200000:
        print("WARNING: Parameter limit exceeded!")
    
    # --- Callbacks ---
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=8,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best_edge_model',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )
    
    # --- Train Model ---
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=False,
        enable_progress_bar=True
    )
    
    print("\nStarting Training...")
    trainer.fit(model, train_loader, val_loader)

    if test_loader is not None:
        print("\nRunning Inference on Test Set...")
        predictions = trainer.predict(model, test_loader)
        print(f"Generated predictions for {sum(len(p) for p in predictions)} test images.")
    
    # --- Export ---
    print("\nStarting Inference Model Export...")
    if checkpoint_callback.best_model_path:
        best_model = TrafficSignLightningModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    else:
        best_model = model
        
    onnx_path = export_to_onnx(best_model.model, INPUT_SHAPE, export_path="edge_ai_traffic_sign.onnx")
    
    # --- Resource Footprint Report ---
    onnx_size_bytes = os.path.getsize(onnx_path)
    onnx_size_kb = onnx_size_bytes / 1024
    
    print("\n" + "="*50)
    print("🚀 EDGE AI FOOTPRINT SUMMARY")
    print("="*50)
    print(f"Target Hardware:       ESP32-S3")
    print(f"Model Architecture:    Separable CNN + Global Average Pooling")
    print(f"Parameter Count:       {param_count:,} out of 200,000 max ({param_count/200000*100:.2f}%)")
    print(f"ONNX Payload Size:     {onnx_size_kb:.2f} KB (Before INT8 Quantization)")
    print("="*50)
    
    print("\nDEPLOYMENT NOTES FOR ESP32-S3:")
    print("- To deploy, convert this ONNX model to TensorFlow Edge/TFLite")
    print("- TFLite Micro will run this network within the internal 512KB SRAM.")

if __name__ == "__main__":
    main()
