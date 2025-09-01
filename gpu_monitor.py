import pynvml
import time

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0, change if multiple GPUs

max_util = 0
max_mem = 0

try:
    while True:
        # GPU utilization %
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        # Memory usage (bytes)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = mem_info.used / 1024**2  # MB

        # Track maximums
        max_util = max(max_util, util)
        max_mem = max(max_mem, mem_used)

        time.sleep(.01)

except KeyboardInterrupt:
    print("\n--- Run Finished ---")
    print(f"Max GPU Utilization: {max_util}%")
    print(f"Max GPU Memory Usage: {max_mem:.2f} MB")

finally:
    pynvml.nvmlShutdown()