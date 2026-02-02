import ctypes
import numpy as np

# Load CUDA Runtime
try:
    libcudart = ctypes.CDLL('libcudart.so')
except OSError:
    try:
        libcudart = ctypes.CDLL('libcudart.so.11')
    except OSError:
        try:
            libcudart = ctypes.CDLL('libcudart.so.12')
        except OSError:
            # Fallback for Windows if needed, though this project is Linux focused
            libcudart = ctypes.CDLL('cudart64_110.dll')

# CUDA Error codes
class cudaError_t:
    cudaSuccess = 0

# CUDA Memcpy kinds
class cudaMemcpyKind:
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    cudaMemcpyDeviceToDevice = 3

def check_cuda_err(err):
    if err != cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Error: {err}")

def cudaMalloc(size):
    ptr = ctypes.c_void_p()
    err = libcudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    return err, ptr.value

def cudaFree(ptr):
    if ptr:
        return libcudart.cudaFree(ctypes.c_void_p(ptr))
    return 0

def cudaMemcpy(dst, src, size, kind):
    return libcudart.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(size), ctypes.c_int(kind)),

def cudaMemset(ptr, value, count):
    return libcudart.cudaMemset(ctypes.c_void_p(ptr), ctypes.c_int(value), ctypes.c_size_t(count))

def cudaStreamCreate():
    stream = ctypes.c_void_p()
    err = libcudart.cudaStreamCreate(ctypes.byref(stream))
    return err, stream.value

def cudaStreamDestroy(stream):
    if stream:
        return libcudart.cudaStreamDestroy(ctypes.c_void_p(stream))
    return 0

def cudaStreamSynchronize(stream):
    return libcudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
