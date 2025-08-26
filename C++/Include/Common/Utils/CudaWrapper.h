/*******************************************************
 文件名：CudaWrapper.h
 作者：sharkls
 描述：CUDA内存管理类，用于管理CUDA内存
 版本：v1.0
 日期：2025-06-18
 *******************************************************/

#ifndef __CUDA_WRAPPER_H__
#define __CUDA_WRAPPER_H__

#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>
#include <vector>


/*
#define checkCudaErrors(status)                                                                                      \
do {                                                                                                               \
    auto ret = (status);                                                                                             \
    if (ret != 0) {                                                                                                  \
    std::cout << "Cuda failure: " << cudaGetErrorString(ret) << " at line " << __LINE__ << " in file " << __FILE__ \
                << " error status: " << ret << "\n";                                                                 \
    abort();                                                                                                       \
    }                                                                                                                \
} while (0)
*/

// 定义错误检查宏： CUDA 函数的返回值不是 cudaSuccess，则打印错误信息并退出程序。
#define checkCudaErrors(val)                                                                     \
  {                                                                                              \
    cudaError_t err = (val);                                                                     \
    if (err != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(err);                                                                                 \
    }                                                                                            \
  }

// 定义模板类 CudaWrapper，用于管理 CUDA 内存。
template <typename T, typename Enable = void>
struct CudaWrapper;

// CudaWrapper 的实现
template <class T>
class CudaWrapper<T, typename std::enable_if_t<std::is_trivial<T>::value && std::is_standard_layout<T>::value, void>> 
{
 public:
  /// @brief Delete default copy constructor.
  CudaWrapper(const CudaWrapper &cudaWrapper) = delete;
  /// @brief Delete default copy assignment operator.
  CudaWrapper &operator=(const CudaWrapper &cudaWrapper) = delete;

  /// @brief Default Constructor.
  /// @param ptr_ Iner CUDA pointer=nullptr.
  CudaWrapper() {
    size_ = 0U;
    capacity_ = 0U;
    ptr_ = nullptr;
  }

  /// @brief Construct a CUDA object with size : malloc memory with size and memset 0.
  CudaWrapper(std::uint64_t size) {
    if (size != 0) {
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void **)&ptr_, getSizeBytes()));
      checkCudaErrors(cudaMemset(ptr_, 0, getSizeBytes()));
    }
  }

  /// @brief Construct a CUDA object with given cpu pointer.
  CudaWrapper(T *cpu_ptr, std::uint64_t size) {
    if (size != 0) {
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void **)&ptr_, getSizeBytes()));
      checkCudaErrors(cudaMemcpy(ptr_, (void *)cpu_ptr, getSizeBytes(), cudaMemcpyHostToDevice));
    }
  }

  /// @brief Construct a CUDA object with given vec_data in cpu and size.
  CudaWrapper(const std::vector<T> &vec_data, std::uint64_t size) {
    if (vec_data.size() >= size) {
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void **)&ptr_, getSizeBytes()));
      cudaMemcpyH2D(vec_data, ptr_, getSizeBytes());
    }
  }

  /// @brief Construct a CUDA object with given vec_data in cpu only.
  CudaWrapper(const std::vector<T> &vec_data) : CudaWrapper(vec_data, vec_data.size()) {}

  /// @brief Moving Construct for improving efficiency in resource management.
  CudaWrapper(CudaWrapper &&cudaWrapper) {
    size_ = cudaWrapper.size_;
    capacity_ = cudaWrapper.capacity_;
    ptr_ = cudaWrapper.ptr_;

    cudaWrapper.size_ = 0U;
    cudaWrapper.capacity_ = 0U;
    cudaWrapper.ptr_ = nullptr;
  }

  /// @brief Moving assignment operator for improving efficiency in resource management.
  CudaWrapper &operator=(CudaWrapper &&cudaWrapper) {
    if (this != &cudaWrapper) {
      size_ = cudaWrapper.size_;
      capacity_ = cudaWrapper.capacity_;
      ptr_ = cudaWrapper.ptr_;

      cudaWrapper.size_ = 0U;
      cudaWrapper.capacity_ = 0U;
      cudaWrapper.ptr_ = nullptr;
    }
    return *this;
  }

  ~CudaWrapper() {
    size_ = 0U;
    capacity_ = 0U;
    if (ptr_ != nullptr) {
      checkCudaErrors(cudaFree(ptr_));
      ptr_ = nullptr;
    }
  }

  /// @brief Copy data to current CUDAobject ptr_ from CudaWrapper's CUDA memory ptr_.
  void cudaMemcpyD2DWrap(const T *src_cuda_ptr, const std::uint64_t size) {
    if (size > capacity_) {
      checkCudaErrors(cudaFree(ptr_));
      size_ = size;
      capacity_ = size;
      checkCudaErrors(cudaMalloc((void **)&ptr_, getSizeBytes()));
    } else {
      size_ = size;
    }
    cudaMemcpyD2D(src_cuda_ptr);
  }

  /// @brief Copy data to current CUDAobject ptr_ from CPU memory(vector), private ptr_ will be changed.
  void cudaMemUpdateWrap(const std::vector<T> &vec_data) {
    if (vec_data.size() > capacity_) {
      checkCudaErrors(cudaFree(ptr_));
      size_ = vec_data.size();
      capacity_ = vec_data.size();
      checkCudaErrors(cudaMalloc((void **)&ptr_, getSizeBytes()));
    } else {
      size_ = vec_data.size();
    }

    cudaMemcpyH2D(vec_data, ptr_, getSizeBytes());
  }

  /// @brief Asynchronous copy data to current CUDAobject ptr_ from CPU memory(vector)
  void cudaMemUpdateWrapAsync(const std::vector<T> &vec_data, cudaStream_t stream) {
    if (vec_data.size() > capacity_) {
      checkCudaErrors(cudaFree(ptr_));
      size_ = vec_data.size();
      capacity_ = vec_data.size();
      checkCudaErrors(cudaMalloc((void **)&ptr_, getSizeBytes()));
    } else {
      size_ = vec_data.size();
    }

    cudaMemcpyH2DAsync(vec_data, ptr_, getSizeBytes(), stream);
  }

  /// @brief Optimized transfer using pinned memory for better performance
  void cudaMemUpdateWrapOptimized(const std::vector<T> &vec_data, cudaStream_t stream) {
    if (vec_data.size() > capacity_) {
      checkCudaErrors(cudaFree(ptr_));
      size_ = vec_data.size();
      capacity_ = vec_data.size();
      checkCudaErrors(cudaMalloc((void **)&ptr_, getSizeBytes()));
    } else {
      size_ = vec_data.size();
    }

    // 使用固定内存进行传输
    void* pinned_host_ptr;
    checkCudaErrors(cudaMallocHost(&pinned_host_ptr, getSizeBytes()));
    
    // 复制数据到固定内存
    memcpy(pinned_host_ptr, vec_data.data(), getSizeBytes());
    
    // 异步传输 - 修复参数顺序
    checkCudaErrors(cudaMemcpyAsync(ptr_, pinned_host_ptr, getSizeBytes(), cudaMemcpyHostToDevice, stream));
    
    // 在传输完成后释放固定内存
    cudaFreeHost(pinned_host_ptr);
  }

  /// @brief Copy data to CPU memory(vector) from current GPU memory(ptr_).
  /// @param
  /// @return std::vector<T>
  std::vector<T> cudaMemcpyD2HResWrap() const {
    if (size_ == 0) {
      return {};
    }

    return cudaMemcpyD2HRes();
  }

  /// @brief Copy data to given CPU memory(vector) from curren GPU memory(ptr_).
  /// @param buf, std::vector<T>
  /// @return
  void cudaMemcpyD2HParamWrap(const std::vector<T> &buf) const {
    if (buf.size() < size_) {
      return;
    }
    cudaMemcpyD2HParam(buf);
  }

  inline void cudaMemSetWrap() { checkCudaErrors(cudaMemset(ptr_, 0, getSizeBytes())); }

  void cudaMemSetWrap(const T &value) {
    std::vector<T> vecdata(size_, value);
    checkCudaErrors(cudaMemcpy(ptr_, (void *)vecdata.data(), getSizeBytes(), cudaMemcpyHostToDevice));
  }

  inline T *getCudaPtr() const { return ptr_; }

  inline std::uint64_t getSize() const { return size_; }

  /// @brief Allocate GPU memory with specified size
  /// @param size Size to allocate
  /// @return true if allocation successful, false otherwise
  bool allocate(std::uint64_t size) {
    if (size == 0) {
      return false;
    }
    
    // Free existing memory if any
    if (ptr_ != nullptr) {
      checkCudaErrors(cudaFree(ptr_));
      ptr_ = nullptr;
    }
    
    size_ = size;
    capacity_ = size;
    
    cudaError_t err = cudaMalloc((void **)&ptr_, getSizeBytes());
    if (err != cudaSuccess) {
      size_ = 0U;
      capacity_ = 0U;
      ptr_ = nullptr;
      return false;
    }
    
    checkCudaErrors(cudaMemset(ptr_, 0, getSizeBytes()));
    return true;
  }

  /// @brief Copy data from host to device
  /// @param host_data Host data pointer
  /// @param offset Offset in device memory
  /// @param size Size to copy
  /// @return true if copy successful, false otherwise
  bool copyFromHost(const void* host_data, std::uint64_t offset, std::uint64_t size) {
    if (ptr_ == nullptr || host_data == nullptr || offset + size > size_) {
      return false;
    }
    
    cudaError_t err = cudaMemcpy((char*)ptr_ + offset * sizeof(T), host_data, 
                                 size * sizeof(T), cudaMemcpyHostToDevice);
    return err == cudaSuccess;
  }

  /// @brief Check if the wrapper is valid (has allocated memory)
  /// @return true if valid, false otherwise
  bool isValid() const {
    return ptr_ != nullptr && size_ > 0;
  }

 private:
  inline std::uint64_t getSizeBytes() const { return size_ * sizeof(std::remove_pointer_t<decltype(ptr_)>); }

  inline void cudaMemcpyD2D(const T *src_cuda_ptr) const {
    checkCudaErrors(cudaMemcpy(ptr_, (void *)src_cuda_ptr, getSizeBytes(), cudaMemcpyDeviceToDevice));
  }

  inline void cudaMemcpyH2D(const std::vector<T> &vec_data, T *cuda_ptr, const std::uint64_t &size) {
    checkCudaErrors(cudaMemcpy(cuda_ptr, (void *)vec_data.data(), size, cudaMemcpyHostToDevice));
  }

  inline void cudaMemcpyH2DAsync(const std::vector<T> &vec_data, T *cuda_ptr, const std::uint64_t &size, cudaStream_t stream) {
    checkCudaErrors(cudaMemcpyAsync(cuda_ptr, (void *)vec_data.data(), size, cudaMemcpyHostToDevice, stream));
  }

  inline std::vector<T> cudaMemcpyD2HRes() const {
    std::vector<T> buf(size_);
    checkCudaErrors(cudaMemcpy((void *)buf.data(), ptr_, getSizeBytes(), cudaMemcpyDeviceToHost));

    return buf;
  }

  inline void cudaMemcpyD2HParam(const std::vector<T> &buf) const {
    checkCudaErrors(cudaMemcpy((void *)buf.data(), ptr_, getSizeBytes(), cudaMemcpyDeviceToHost));
  }

 private:
  std::uint64_t size_ = 0U;
  std::uint64_t capacity_ = 0U;
  T *ptr_ = nullptr;
};


#endif  // __CUDA_WRAPPER_H__
