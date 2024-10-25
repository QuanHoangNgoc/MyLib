import os
import sys
import torch
import psutil
import gc


class ut:
    def __init__(self) -> None:
        pass


class ut(ut):
    def check_ram():
        # Check system RAM usage
        ram_info = psutil.virtual_memory()
        # Extract and print RAM usage stats in bytes
        print(f"Used RAM: {ram_info.used / (1024 ** 3):.2f} GB")
        print(f"RAM Usage Percentage: {ram_info.percent}%")

    def check_vram():
        # Check current VRAM usage (in bytes)
        for device_id in range(torch.cuda.device_count()):
            current_vram = torch.cuda.memory_allocated(device_id)
            total_vram = torch.cuda.get_device_properties(device_id).total_memory
            print(
                f"GPU {device_id} - Current VRAM Usage: {current_vram / (1024 ** 2):.2f} MB"
            )
            print(
                f"GPU {device_id} - VRAM Usage Percentage: {current_vram / total_vram * 100:.2f}%"
            )

    def clear_space():
        gc.collect()  # Clear unused objects
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("Clear space... ...")

    def tensor_mb(x):
        tensor_memory = x.element_size() * x.numel()
        print(f"Tensor Memory: {tensor_memory / (1024 ** 2):.2f} MB")
        print(f"Tensor Device: {x.device}")
        print(f"Tensor Data Type: {x.dtype}")
        print(f"Requires Grad: {x.requires_grad}")
        print(f"Shape: {x.shape}")

    def model_mb(model):
        try:
            model.print_trainable_parameters()
        except Exception as e:
            print(e)
        model_memory = sum(
            param.element_size() * param.numel() for param in model.parameters()
        ) + sum(buffer.element_size() * buffer.numel() for buffer in model.buffers())
        print(f"Model Memory: {model_memory / (1024 ** 2):.2f} MB")
        params = list(model.named_parameters())
        top_n = 2
        for name, param in params[:top_n] + params[-top_n:]:
            print("... ....")
            print(f"Name: {name}")
            print(f"  Device: {param.device}")
            print(f"  Data Type: {param.dtype}")
            print(f"  Requires Grad: {param.requires_grad}")

    def mess(mess, lv=0):
        if lv < 0:
            return
        mess = mess[0].upper() + mess[1:].lower()
        for x in ["!"]:
            mess = mess.replace(x, "")
        if lv == 0:
            print(f"[+]->mess: {mess}")
        if lv > 0:
            print(f"[+]{'-'*lv}level={lv}->mess: {mess}")


if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    ut.check_ram()
    ut.check_vram()
    ut.mess("Hi Quan", 0)
    ut.mess("Hi Quan", 1)
