#include "cuda_generate.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <omp.h>
#include <vector>
#include <mutex>

// 错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// 配置参数
#define MAX_BATCH_SIZE 20000
#define MAX_STRING_LENGTH 128
#define NUM_STREAMS 8
#define BLOCK_SIZE 256
#define ALIGNMENT 512
#define MIN_PARALLEL_SIZE 5000  // 最小并行处理阈值

// 全局互斥锁保护CUDA初始化
static std::mutex cuda_init_mutex;
static bool cuda_initialized = false;

// CUDA初始化函数
void initializeCUDA() {
    std::lock_guard<std::mutex> lock(cuda_init_mutex);
    if (!cuda_initialized) {
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            std::cerr << "No CUDA-capable devices found!" << std::endl;
            exit(1);
        }
        
        // 设置设备并预热
        CUDA_CHECK(cudaSetDevice(0));
        void* temp;
        CUDA_CHECK(cudaMalloc(&temp, 1));
        CUDA_CHECK(cudaFree(temp));
        cuda_initialized = true;
        
        std::cout << "CUDA initialized with " << deviceCount << " device(s)" << std::endl;
    }
}

// 优化的单段核函数
__global__ void optimizedSingleSegmentKernel(const char* __restrict__ input, 
                                             char* __restrict__ output,
                                             const int* __restrict__ offsets,
                                             const int* __restrict__ lengths,
                                             int num_strings) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_strings) return;
    
    const int offset = offsets[idx];
    const int len = lengths[idx];
    const char* src = input + offset;
    char* dst = output + offset;
    
    // 简单逐字节复制
    for (int i = 0; i < len; ++i) {
        dst[i] = src[i];
    }
    
    // 添加null终止符
    if (len > 0) dst[len] = '\0';
}

// 常量内存存储前缀
__constant__ char d_prefix[MAX_STRING_LENGTH];
__constant__ int d_prefix_length;

// 优化的多段核函数
__global__ void optimizedMultiSegmentKernel(const char* __restrict__ input, 
                                           char* __restrict__ output,
                                           const int* __restrict__ input_offsets,
                                           const int* __restrict__ output_offsets,
                                           const int* __restrict__ lengths,
                                           int num_strings) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_strings) return;
    
    const int prefix_len = d_prefix_length;
    const int str_len = lengths[idx];
    
    // 输出位置
    char* dst = output + output_offsets[idx];
    
    // 复制前缀
    for (int i = 0; i < prefix_len; ++i) {
        dst[i] = d_prefix[i];
    }
    
    // 复制后缀
    const char* src = input + input_offsets[idx];
    for (int i = 0; i < str_len; ++i) {
        dst[prefix_len + i] = src[i];
    }
    
    // 添加null终止符
    dst[prefix_len + str_len] = '\0';
}

// 对齐的内存分配函数
void* alignedMalloc(size_t size) {
    void* ptr = nullptr;
    #ifdef _WIN32
        ptr = _aligned_malloc(size, ALIGNMENT);
    #else
        if (posix_memalign(&ptr, ALIGNMENT, size) != 0) {
            ptr = nullptr;
        }
    #endif
    return ptr;
}

// 对齐的内存释放函数
void alignedFree(void* ptr) {
    #ifdef _WIN32
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

// 带CUDA流的单段生成GPU实现（修复版本）
void generateSingleSegmentGPU_WithStream(const std::vector<std::string>& values,
                                        std::vector<std::string>& guesses,
                                        int& total_guesses,
                                        cudaStream_t stream)
{
    if (values.empty()) return;
    const size_t num_values = values.size();
    
    // 在设备上分配内存 - 使用传统的cudaMalloc
    char *d_input, *d_output;
    int *d_offsets, *d_lengths;
    
    // 计算总内存需求 - 确保对齐
    size_t total_mem = 0;
    std::vector<int> h_offsets(num_values);
    std::vector<int> h_lengths(num_values);
    
    // 计算对齐偏移
    for (size_t i = 0; i < num_values; ++i) {
        h_lengths[i] = values[i].size();
        // 确保每个字符串对齐
        h_offsets[i] = (total_mem + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        total_mem = h_offsets[i] + h_lengths[i] + 1;
    }
    
    // 分配对齐的主机内存
    char* h_input = static_cast<char*>(alignedMalloc(total_mem));
    if (!h_input) {
        std::cerr << "Failed to allocate aligned host memory" << std::endl;
        return;
    }
    
    // 分配设备内存 - 使用传统的cudaMalloc
    CUDA_CHECK(cudaMalloc(&d_input, total_mem));
    CUDA_CHECK(cudaMalloc(&d_output, total_mem));
    CUDA_CHECK(cudaMalloc(&d_offsets, num_values * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lengths, num_values * sizeof(int)));
    
    // 填充主机内存缓冲区
    for (size_t i = 0; i < num_values; ++i) {
        memcpy(h_input + h_offsets[i], values[i].c_str(), h_lengths[i] + 1);
    }
    
    // 异步数据传输
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, total_mem, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_offsets, h_offsets.data(), num_values * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_lengths, h_lengths.data(), num_values * sizeof(int), cudaMemcpyHostToDevice, stream));
    
    // 启动核函数
    const int grid_size = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;
    optimizedSingleSegmentKernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_input, d_output, d_offsets, d_lengths, num_values);
    
    // 回传结果
    CUDA_CHECK(cudaMemcpyAsync(h_input, d_output, total_mem, cudaMemcpyDeviceToHost, stream));
    
    // 同步流
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // 构建结果字符串
    size_t original_size = guesses.size();
    guesses.resize(original_size + num_values);
    for (size_t i = 0; i < num_values; ++i) {
        guesses[original_size + i] = std::string(h_input + h_offsets[i]);
    }
    
    // 更新计数器
    total_guesses += num_values;
    
    // 清理资源 - 使用传统的cudaFree
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_lengths));
    alignedFree(h_input);
}

// 带CUDA流的多段生成GPU实现（修复版本）
void generateMultiSegmentGPU_WithStream(const std::string& prefix,
                                       const std::vector<std::string>& values,
                                       std::vector<std::string>& guesses,
                                       int& total_guesses,
                                       cudaStream_t stream)
{
    if (values.empty()) return;
    const size_t num_values = values.size();
    const int prefix_len = prefix.length();
    
    // 检查长度限制
    if (prefix_len >= MAX_STRING_LENGTH) {
        std::cerr << "Prefix too long: " << prefix_len << std::endl;
        return;
    }
    
    // 将前缀复制到常量内存
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_prefix, prefix.c_str(), prefix_len + 1, 0, cudaMemcpyHostToDevice, stream));
    int prefix_length_val = prefix_len;
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_prefix_length, &prefix_length_val, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
    
    // 在设备上分配内存 - 使用传统的cudaMalloc
    char *d_input, *d_output;
    int *d_input_offsets, *d_output_offsets, *d_lengths;
    
    // 准备主机数据
    std::vector<int> h_input_offsets(num_values);
    std::vector<int> h_output_offsets(num_values);
    std::vector<int> h_lengths(num_values);
    
    size_t input_mem = 0;
    const size_t output_mem = num_values * MAX_STRING_LENGTH;
    
    // 对齐输入偏移量
    for (size_t i = 0; i < num_values; ++i) {
        h_lengths[i] = values[i].size();
        h_input_offsets[i] = (input_mem + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        input_mem = h_input_offsets[i] + h_lengths[i] + 1;
        h_output_offsets[i] = i * MAX_STRING_LENGTH;
    }
    
    // 分配主机内存
    char* h_input = static_cast<char*>(alignedMalloc(input_mem));
    char* h_output = static_cast<char*>(malloc(output_mem));
    if (!h_input || !h_output) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        if (h_input) alignedFree(h_input);
        if (h_output) free(h_output);
        return;
    }
    
    // 分配设备内存 - 使用传统的cudaMalloc
    CUDA_CHECK(cudaMalloc(&d_input, input_mem));
    CUDA_CHECK(cudaMalloc(&d_output, output_mem));
    CUDA_CHECK(cudaMalloc(&d_input_offsets, num_values * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_offsets, num_values * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lengths, num_values * sizeof(int)));
    
    // 填充主机输入缓冲区
    for (size_t i = 0; i < num_values; ++i) {
        memcpy(h_input + h_input_offsets[i], values[i].c_str(), h_lengths[i] + 1);
    }
    
    // 异步数据传输
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, input_mem, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_input_offsets, h_input_offsets.data(), num_values * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_output_offsets, h_output_offsets.data(), num_values * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_lengths, h_lengths.data(), num_values * sizeof(int), cudaMemcpyHostToDevice, stream));
    
    // 启动核函数
    const int grid_size = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;
    optimizedMultiSegmentKernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, d_input_offsets, d_output_offsets, d_lengths, num_values
    );
    
    // 回传结果
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, output_mem, cudaMemcpyDeviceToHost, stream));
    
    // 同步流
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // 构建结果字符串
    size_t original_size = guesses.size();
    guesses.resize(original_size + num_values);
    for (size_t i = 0; i < num_values; ++i) {
        guesses[original_size + i] = std::string(h_output + h_output_offsets[i]);
    }
    
    // 更新计数器
    total_guesses += num_values;
    
    // 清理资源 - 使用传统的cudaFree
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input_offsets));
    CUDA_CHECK(cudaFree(d_output_offsets));
    CUDA_CHECK(cudaFree(d_lengths));
    alignedFree(h_input);
    free(h_output);
}

// 并行单段生成GPU实现
void generateSingleSegmentGPU_Parallel(const std::vector<std::string>& values,
                                      std::vector<std::string>& guesses,
                                      int& total_guesses,
                                      int num_threads)
{
    if (values.empty()) return;
    
    // 初始化CUDA
    initializeCUDA();
    
    // 对于小数据集，直接使用单线程GPU处理
    if (values.size() < MIN_PARALLEL_SIZE) {
        generateSingleSegmentGPU(values, guesses, total_guesses);
        return;
    }
    
    const size_t total_values = values.size();
    const size_t chunk_size = (total_values + num_threads - 1) / num_threads;
    
    // 为每个线程准备结果容器
    std::vector<std::vector<std::string>> thread_results(num_threads);
    std::vector<int> thread_counts(num_threads, 0);
    
    // 预分配结果容器大小
    for (int i = 0; i < num_threads; ++i) {
        size_t start_idx = i * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, total_values);
        if (start_idx < end_idx) {
            thread_results[i].reserve(end_idx - start_idx);
        }
    }
    
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        size_t start_idx = thread_id * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, total_values);
        
        if (start_idx < end_idx) {
            // 创建该线程的数据子集
            std::vector<std::string> thread_values(
                values.begin() + start_idx, 
                values.begin() + end_idx
            );
            
            // 每个线程使用不同的CUDA流
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            
            try {
                // 调用GPU处理该线程的数据块
                generateSingleSegmentGPU_WithStream(thread_values, 
                                                   thread_results[thread_id], 
                                                   thread_counts[thread_id],
                                                   stream);
            } catch (const std::exception& e) {
                std::cerr << "Thread " << thread_id << " error: " << e.what() << std::endl;
            }
            
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
    }
    
    // 合并所有线程的结果
    size_t total_result_size = 0;
    for (int i = 0; i < num_threads; ++i) {
        total_result_size += thread_results[i].size();
    }
    
    guesses.reserve(guesses.size() + total_result_size);
    for (int i = 0; i < num_threads; ++i) {
        guesses.insert(guesses.end(), 
                      thread_results[i].begin(), 
                      thread_results[i].end());
        total_guesses += thread_counts[i];
    }
}

// 并行多段生成GPU实现
void generateMultiSegmentGPU_Parallel(const std::string& prefix,
                                     const std::vector<std::string>& values,
                                     std::vector<std::string>& guesses,
                                     int& total_guesses,
                                     int num_threads)
{
    if (values.empty()) return;
    
    // 初始化CUDA
    initializeCUDA();
    
    // 对于小数据集，直接使用单线程GPU处理
    if (values.size() < MIN_PARALLEL_SIZE) {
        generateMultiSegmentGPU(prefix, values, guesses, total_guesses);
        return;
    }
    
    const size_t total_values = values.size();
    const size_t chunk_size = (total_values + num_threads - 1) / num_threads;
    
    // 为每个线程准备结果容器
    std::vector<std::vector<std::string>> thread_results(num_threads);
    std::vector<int> thread_counts(num_threads, 0);
    
    // 预分配结果容器大小
    for (int i = 0; i < num_threads; ++i) {
        size_t start_idx = i * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, total_values);
        if (start_idx < end_idx) {
            thread_results[i].reserve(end_idx - start_idx);
        }
    }
    
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        size_t start_idx = thread_id * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, total_values);
        
        if (start_idx < end_idx) {
            // 创建该线程的数据子集
            std::vector<std::string> thread_values(
                values.begin() + start_idx, 
                values.begin() + end_idx
            );
            
            // 每个线程使用不同的CUDA流
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            
            try {
                // 调用GPU处理该线程的数据块
                generateMultiSegmentGPU_WithStream(prefix,
                                                  thread_values, 
                                                  thread_results[thread_id], 
                                                  thread_counts[thread_id],
                                                  stream);
            } catch (const std::exception& e) {
                std::cerr << "Thread " << thread_id << " error: " << e.what() << std::endl;
            }
            
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
    }
    
    // 合并所有线程的结果
    size_t total_result_size = 0;
    for (int i = 0; i < num_threads; ++i) {
        total_result_size += thread_results[i].size();
    }
    
    guesses.reserve(guesses.size() + total_result_size);
    for (int i = 0; i < num_threads; ++i) {
        guesses.insert(guesses.end(), 
                      thread_results[i].begin(), 
                      thread_results[i].end());
        total_guesses += thread_counts[i];
    }
}

// 保持原有的函数实现（修复版本）
void generateSingleSegmentGPU(const std::vector<std::string>& values,
                             std::vector<std::string>& guesses,
                             int& total_guesses)
{
    // 初始化CUDA
    initializeCUDA();
    
    if (values.empty()) return;
    
    // 创建CUDA流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    try {
        generateSingleSegmentGPU_WithStream(values, guesses, total_guesses, stream);
    } catch (const std::exception& e) {
        std::cerr << "GPU processing error: " << e.what() << std::endl;
    }
    
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void generateMultiSegmentGPU(const std::string& prefix,
                            const std::vector<std::string>& values,
                            std::vector<std::string>& guesses,
                            int& total_guesses)
{
    // 初始化CUDA
    initializeCUDA();
    
    if (values.empty()) return;
    
    // 创建CUDA流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    try {
        generateMultiSegmentGPU_WithStream(prefix, values, guesses, total_guesses, stream);
    } catch (const std::exception& e) {
        std::cerr << "GPU processing error: " << e.what() << std::endl;
    }
    
    CUDA_CHECK(cudaStreamDestroy(stream));
}