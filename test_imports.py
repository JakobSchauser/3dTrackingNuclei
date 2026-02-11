#!/usr/bin/env python3

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    import cellpose
    print(f"Cellpose imported successfully")

    import ultrack
    print(f"Ultrack imported successfully")

    print("All packages imported successfully!")

except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Other error: {e}")