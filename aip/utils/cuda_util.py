import torch


def check_cuda() -> bool:

    print(f"PyTorch version:   {torch.__version__}")
    print(f"CUDA available:    {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version:      {torch.version.cuda}")
        print(f"Device count:      {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")

        # Quick tensor op on GPU
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"\nTest tensor on GPU: {x}")
        print("✅ CUDA is working!")
        return True
    else:
        print("❌ CUDA is NOT available!")
        return False
