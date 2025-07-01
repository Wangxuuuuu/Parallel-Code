#include "cuda_generate.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdlib>

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
#define ALIGNMENT 512  // 增加对齐大小到512字节

// 优化的单段核函数 - 简化版本避免对齐问题
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

// 优化的多段核函数 - 使用常量内存存储前缀
__constant__ char d_prefix[MAX_STRING_LENGTH];
__constant__ int d_prefix_length;

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

// 优化的单段生成GPU实现
void generateSingleSegmentGPU(const std::vector<std::string>& values, 
                             std::vector<std::string>& guesses, 
                             int& total_guesses) 
{
    if (values.empty()) return;
    const size_t num_values = values.size();
    
    // 创建CUDA流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // 在设备上分配内存
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
    
    // 分配设备内存
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
    
    // 同步
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // 构建结果字符串
    guesses.reserve(guesses.size() + num_values);
    for (size_t i = 0; i < num_values; ++i) {
        guesses.emplace_back(h_input + h_offsets[i]);
    }
    
    // 更新计数器
    total_guesses += num_values;
    
    // 清理资源
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_lengths));
    alignedFree(h_input);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// 优化的多段生成GPU实现
void generateMultiSegmentGPU(const std::string& prefix, 
                           const std::vector<std::string>& values, 
                           std::vector<std::string>& guesses, 
                           int& total_guesses) 
{
    if (values.empty()) return;
    const size_t num_values = values.size();
    const int prefix_len = prefix.length();
    
    // 检查长度限制
    if (prefix_len >= MAX_STRING_LENGTH) {
        std::cerr << "Prefix too long: " << prefix_len << std::endl;
        return;
    }
    
    // 创建CUDA流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // 将前缀复制到常量内存
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_prefix, prefix.c_str(), prefix_len + 1, 0, cudaMemcpyHostToDevice, stream));
    int prefix_length_val = prefix_len;
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_prefix_length, &prefix_length_val, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
    
    // 在设备上分配内存
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
        // 确保每个字符串对齐
        h_input_offsets[i] = (input_mem + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        input_mem = h_input_offsets[i] + h_lengths[i] + 1;
        h_output_offsets[i] = i * MAX_STRING_LENGTH;
    }
    
    // 分配主机内存
    char* h_input = static_cast<char*>(alignedMalloc(input_mem));
    if (!h_input) {
        std::cerr << "Failed to allocate aligned host memory" << std::endl;
        return;
    }
    char* h_output = static_cast<char*>(malloc(output_mem));
    
    // 分配设备内存
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
    
    // 同步
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // 构建结果字符串
    guesses.reserve(guesses.size() + num_values);
    for (size_t i = 0; i < num_values; ++i) {
        guesses.emplace_back(h_output + h_output_offsets[i]);
    }
    
    // 更新计数器
    total_guesses += num_values;
    
    // 清理资源
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input_offsets));
    CUDA_CHECK(cudaFree(d_output_offsets));
    CUDA_CHECK(cudaFree(d_lengths));
    alignedFree(h_input);
    free(h_output);
    CUDA_CHECK(cudaStreamDestroy(stream));
}