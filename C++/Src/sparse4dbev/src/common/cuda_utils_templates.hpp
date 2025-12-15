#ifndef __CUDA_UTILS_TEMPLATES_HPP__
#define __CUDA_UTILS_TEMPLATES_HPP__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace sparse4d {
namespace common {

// Helper trait to convert any type to float (for computation)
template <typename T>
__device__ __forceinline__ float val_to_float(T val);

template <>
__device__ __forceinline__ float val_to_float<float>(float val) {
    return val;
}

template <>
__device__ __forceinline__ float val_to_float<half>(half val) {
    return __half2float(val);
}

// Helper trait to convert float result back to T (for storage)
template <typename T>
__device__ __forceinline__ T float_to_val(float val);

template <>
__device__ __forceinline__ float float_to_val<float>(float val) {
    return val;
}

template <>
__device__ __forceinline__ half float_to_val<half>(float val) {
    return __float2half(val);
}

// Helper for atomicAdd which might not be supported for half on all archs
template <typename T>
__device__ __forceinline__ void atomic_add(T* address, T val);

template <>
__device__ __forceinline__ void atomic_add<float>(float* address, float val) {
    atomicAdd(address, val);
}

// For half, atomicAdd is supported on newer archs, but let's be safe or use built-in if available
// Assuming compute capability >= 7.0 for FP16 atomicAdd
template <>
__device__ __forceinline__ void atomic_add<half>(half* address, half val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(reinterpret_cast<half*>(address), val);
#else
    // Fallback or error for older cards if needed, but modern auto-driving SOCs support it.
    // Simple CAS loop implementation if strictly necessary
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        half h_old = (size_t)address & 2 ? __ushort_as_half(old >> 16) : __ushort_as_half(old & 0xffff);
        half h_new = h_old + val;
        unsigned int new_val = (size_t)address & 2 ? (old & 0xffff) | (__half_as_ushort(h_new) << 16) : (old & 0xffff0000) | __half_as_ushort(h_new);
        old = atomicCAS(address_as_ui, assumed, new_val);
    } while (assumed != old);
#endif
}

} // namespace common
} // namespace sparse4d

#endif // __CUDA_UTILS_TEMPLATES_HPP__

