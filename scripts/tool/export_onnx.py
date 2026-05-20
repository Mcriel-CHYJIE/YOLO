"""
Export YOLOv8 PyTorch models to ONNX format.

Usage:
    python export_onnx.py

This script exports both fall detection and bed detection models
to ONNX format with 640x640 input size and opset 12.
"""
from pathlib import Path

from ultralytics import YOLO


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Model configuration
MODELS = [
    {
        "pt_path": PROJECT_ROOT.parent / "models" / "best5.20.pt",  # Use latest trained model
        "onnx_path": PROJECT_ROOT.parent / "models" / "best.onnx",
    },

    # {
    #     "pt_path": PROJECT_ROOT.parent / "models" / "yolov8n-fall.pt",
    #     "onnx_path": PROJECT_ROOT.parent / "models" / "yolov8n-fall.onnx",
    # },

    #{
    #    "pt_path": PROJECT_ROOT.parent / "yolo11n.pt",  # Pretrained model in project root
    #    "onnx_path": PROJECT_ROOT.parent / "yolo11n.onnx",
    #},
]

# Export settings
IMGSZ = 640
OPSET = 12


def export_model(pt_path: Path, onnx_path: Path, imgsz: int, opset: int) -> bool:
    """
    Export a PyTorch model to ONNX format.

    Args:
        pt_path: Path to the PyTorch model file.
        onnx_path: Path for the output ONNX file.
        imgsz: Input image size (width and height).
        opset: ONNX opset version.

    Returns:
        True if export successful, False otherwise.
    """
    if not pt_path.exists():
        print(f"Error: Model not found: {pt_path}")
        return False

    print(f"Exporting: {pt_path.name} -> {onnx_path.name}")
    print(f"  Input size: {imgsz}x{imgsz}")
    print(f"  Opset: {opset}")

    # Load model
    model = YOLO(str(pt_path))

    # Export to ONNX
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,
        dynamic=False,
    )

    # ultralytics exports to the same directory as the .pt file
    # Move to desired location if different
    default_onnx = pt_path.with_suffix(".onnx")
    if default_onnx != onnx_path and default_onnx.exists():
        default_onnx.rename(onnx_path)

    if onnx_path.exists():
        print(f"  Success: {onnx_path}")
        return True
    else:
        print(f"  Failed: ONNX file not created")
        return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("YOLOv8 PyTorch to ONNX Export")
    print("=" * 60)

    success_count = 0
    for model_config in MODELS:
        pt_path = model_config["pt_path"]
        onnx_path = model_config["onnx_path"]

        print()
        if export_model(pt_path, onnx_path, IMGSZ, OPSET):
            success_count += 1

    print()
    print("=" * 60)
    print(f"Export complete: {success_count}/{len(MODELS)} models exported")
    print("=" * 60)


if __name__ == "__main__":
    main()
