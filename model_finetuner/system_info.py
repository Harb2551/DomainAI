import torch
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None

def get_system_info():
    gpu_info = []
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs() if GPUtil else []
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': f"{gpu.memoryTotal}MB",
                    'memory_used': f"{gpu.memoryUsed}MB",
                    'memory_util': f"{gpu.memoryUtil*100:.1f}%",
                    'temperature': f"{gpu.temperature}°C"
                })
        except:
            gpu_info.append({
                'name': torch.cuda.get_device_name(0),
                'count': torch.cuda.device_count()
            })
    return {
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count(),
        'gpu_info': gpu_info,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'torch_version': torch.__version__,
        'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2)
    }
