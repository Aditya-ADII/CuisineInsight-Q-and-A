import pynvml
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total GPU memory: {mem_info.total / 1024**2:.2f} MB")
    print(f"Free GPU memory: {mem_info.free / 1024**2:.2f} MB")
    print(f"Used GPU memory: {mem_info.used / 1024**2:.2f} MB")
except pynvml.NVMLError as e:
    print(f"NVML Error: {e}")
finally:
    pynvml.nvmlShutdown()