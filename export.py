import torch

def export_to_onnx(model, input_shape, export_path="edge_ai_traffic_sign.onnx"):
    """
    Exports the PyTorch model to ONNX.
    For ESP32-S3, one can then convert ONNX -> TensorFlow -> TFLite (INT8).
    """
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model successfully saved to: {export_path}")
    return export_path
