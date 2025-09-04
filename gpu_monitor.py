import pynvml
import time

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

max_util = 0
max_mem = 0

try:
    while True:
        try:
            # GPU utilization %
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            # Memory usage (bytes)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = mem_info.used / 1024**2  # MB

            # Track maximums
            max_util = max(max_util, util)
            max_mem = max(max_mem, mem_used)

        except pynvml.NVMLError as e:
            # Happens when GPU is idle or process just ended
            util = 0
            mem_used = 0

        time.sleep(0.1)  # slower polling is usually enough

except KeyboardInterrupt:
    print("\n--- Run Finished ---")
    print(pynvml.nvmlDeviceGetName(handle))
    print(f"Max GPU Utilization: {max_util}%")
    print(f"Max GPU Memory Usage: {max_mem:.2f} MB")

finally:
    pynvml.nvmlShutdown()
