#ifndef CUDA_GENERATE_H
#define CUDA_GENERATE_H

#include <vector>
#include <string>
#include <cuda_runtime.h>  // 添加这行来包含CUDA类型定义

// 原有的GPU函数
void generateSingleSegmentGPU(const std::vector<std::string>& values,
                             std::vector<std::string>& guesses,
                             int& total_guesses);

void generateMultiSegmentGPU(const std::string& prefix,
                            const std::vector<std::string>& values,
                            std::vector<std::string>& guesses,
                            int& total_guesses);

// 新增的并行GPU函数
void generateSingleSegmentGPU_Parallel(const std::vector<std::string>& values,
                                      std::vector<std::string>& guesses,
                                      int& total_guesses,
                                      int num_threads = 4);

void generateMultiSegmentGPU_Parallel(const std::string& prefix,
                                     const std::vector<std::string>& values,
                                     std::vector<std::string>& guesses,
                                     int& total_guesses,
                                     int num_threads = 4);

// 内部辅助函数
void generateSingleSegmentGPU_WithStream(const std::vector<std::string>& values,
                                        std::vector<std::string>& guesses,
                                        int& total_guesses,
                                        cudaStream_t stream);

void generateMultiSegmentGPU_WithStream(const std::string& prefix,
                                       const std::vector<std::string>& values,
                                       std::vector<std::string>& guesses,
                                       int& total_guesses,
                                       cudaStream_t stream);

#endif // CUDA_GENERATE_H
