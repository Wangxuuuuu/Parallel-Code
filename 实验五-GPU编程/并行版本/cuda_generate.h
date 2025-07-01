#ifndef CUDA_GENERATE_H
#define CUDA_GENERATE_H

#include <vector>
#include <string>

void generateSingleSegmentGPU(const std::vector<std::string>& values, 
                             std::vector<std::string>& guesses, 
                             int& total_guesses);

void generateMultiSegmentGPU(const std::string& prefix, 
                            const std::vector<std::string>& values, 
                            std::vector<std::string>& guesses, 
                            int& total_guesses);

#endif // CUDA_GENERATE_H
