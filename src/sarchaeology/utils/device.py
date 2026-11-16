"""CUDA smoke test for the RTX 5070 Ti."""

def main() -> None:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: install a matching CUDA-enabled PyTorch wheel and update NVIDIA drivers")
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(512, 512, device="cuda")
    torch.cuda.synchronize()
    print("CUDA matmul OK:", tuple((x @ x).shape))

if __name__ == "__main__":
    main()
