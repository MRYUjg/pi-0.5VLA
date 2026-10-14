"""Host NVML byte-level sampling, including process launch and cold loading."""
import ctypes
import json
import threading
import time


class Memory(ctypes.Structure):
    _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]


class Process(ctypes.Structure):
    _fields_=[('pid',ctypes.c_uint),('usedGpuMemory',ctypes.c_ulonglong),
              ('gpuInstanceId',ctypes.c_uint),('computeInstanceId',ctypes.c_uint)]


class Monitor:
    def __init__(self,path):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1')
        assert self.lib.nvmlInit_v2()==0
        self.handle=ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0),ctypes.byref(self.handle))==0
        self.samples=[];self.errors=[];self.stop=threading.Event()
        self.log=path.open('x',buffering=1)
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def sample(self):
        m=Memory()
        assert self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(m))==0
        n=ctypes.c_uint(64);p=(Process*64)()
        rc=self.lib.nvmlDeviceGetComputeRunningProcesses_v2(self.handle,ctypes.byref(n),p)
        assert rc==0,('NVML compute process query',rc,n.value)
        return {'monotonic_seconds':time.monotonic(),'device_used_bytes':m.used,'device_total_bytes':m.total,
            'processes':{str(v.pid):v.usedGpuMemory for v in p[:n.value]}}
    def loop(self):
        try:
            while not self.stop.is_set():
                row=self.sample();self.samples.append(row);self.log.write(json.dumps(row)+'\n')
                self.stop.wait(.02)
        except Exception as exc:self.errors.append(repr(exc))
    def close(self):
        self.stop.set();self.thread.join()
        row=self.sample();self.samples.append(row);self.log.write(json.dumps(row)+'\n');self.log.close()
        self.lib.nvmlShutdown()
        assert not self.errors,self.errors
        return self.samples
