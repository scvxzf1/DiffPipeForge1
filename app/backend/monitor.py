import sys
import time
import json
import psutil
import traceback

# Try to import pynvml for GPU stats
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

def get_gpu_stats():
    gpus = []
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                temp = 0
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    pass

                gpus.append({
                    "id": i,
                    "name": name,
                    "gpu_util": util.gpu,
                    "mem_total": mem.total,
                    "mem_used": mem.used,
                    "mem_free": mem.free,
                    "temperature": temp
                })
        except Exception as e:
            # Silently fail or log if needed, but don't crash
            pass
    return gpus

import platform

def get_cpu_model():
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return info.get('brand_raw', platform.processor())
    except ImportError:
        return platform.processor() or "Unknown CPU"
    except Exception:
        return platform.processor() or "Unknown CPU"

def main():
    print("Starting Resource Monitor...", file=sys.stderr)
    cpu_model = get_cpu_model()
    
    while True:
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # RAM
            mem = psutil.virtual_memory()
            
            # GPU
            gpu_stats = get_gpu_stats()
            
            # Disk
            disks = []
            try:
                # all=False to ignore internal/virtual stuff on Windows
                partitions = psutil.disk_partitions(all=False)
                for partition in partitions:
                    try:
                        if 'cdrom' in partition.opts or partition.fstype == '':
                            continue
                        usage = psutil.disk_usage(partition.mountpoint)
                        disks.append({
                            "device": partition.device,
                            "mountpoint": partition.mountpoint,
                            "total": usage.total,
                            "used": usage.used,
                            "free": usage.free,
                            "percent": usage.percent
                        })
                    except (PermissionError, OSError):
                        # Catch drive not ready or permission issues for specific drive
                        continue
            except Exception:
                pass

            stats = {
                "cpu_model": cpu_model,
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": mem.total,
                    "available": mem.available,
                    "percent": mem.percent,
                    "used": mem.used
                },
                "disks": disks,
                "gpus": gpu_stats,
                "timestamp": time.time()
            }
            
            # Output as JSON line
            print(f"__JSON_START__{json.dumps(stats)}__JSON_END__", flush=True)
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            err = {"error": str(e)}
            print(f"__JSON_START__{json.dumps(err)}__JSON_END__", flush=True)
            time.sleep(1)

if __name__ == "__main__":
    main()
