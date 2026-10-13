"""Report runtime readiness without launching training."""
import importlib
import sys

print("Python:", sys.version.replace("\n", " "))
for name in ("numpy", "pandas", "sklearn", "joblib", "torch", "torch_geometric"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}: OK", getattr(mod, "__version__", ""))
    except Exception as exc:
        print(f"{name}: MISSING/FAILED ({type(exc).__name__}: {exc})")

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
except Exception:
    pass
