#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
using namespace std;
using namespace chrono;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 计时变量
    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    int history = 0;
    int total_generated = 0;
    int generate_n = 10000000; // 生成上限
    
    // 所有进程都需要模型数据
    model global_model;
    auto start_train = system_clock::now();
    
    if (rank == 0) {
        cout << "Training model with " << size << " MPI processes" << endl;
    }
    
    // 训练模型
    global_model.train("/guessdata/Rockyou-singleLined-full.txt");
    global_model.order();
    
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    
    if (rank == 0) {
        cout << "Model training completed in " << time_train << " seconds" << endl;
    }
    
    // 创建优先级队列并初始化
    PriorityQueue q;
    q.m = global_model;
    
    q.init();
    
    if (rank == 0) {
        cout << "Priority queue initialized with " << q.priority.size() << " PTs" << endl;
    }
    
    auto start = system_clock::now();
    int curr_num = 0;
    
    // 主循环
    bool should_continue = true;
    while (should_continue && !q.priority.empty())
    {
        // 所有进程处理相同的PT序列
        q.PopNext();
        
        // 只有主进程维护guesses列表
        if (rank == 0) {
            total_generated = history + q.guesses.size();
            
            // 进度报告
            if (total_generated - curr_num >= 100000)
            {
                cout << "Guesses generated: " << total_generated << endl;
                curr_num = total_generated;
            }
            
            // 检查是否达到生成上限
            if (total_generated >= generate_n) {
                should_continue = false;
                cout << "Reached generation limit of " << generate_n << " guesses" << endl;
            }
            
            // 定期哈希并清空缓存（即使未达到100万也处理）
            if (!q.guesses.empty() && (q.guesses.size() >= 1000000 || !should_continue)) 
            {
                auto start_hash = system_clock::now();
                bit32 state[4];
                
                for (string pw : q.guesses)
                {
                    MD5Hash(pw, state);
                }
                
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
                
                history += q.guesses.size();
                total_generated = history;
                q.guesses.clear();
                
                if (rank == 0) {
                    cout << "Hashed " << history << " guesses so far" << endl;
                }
            }
        }
        
        // 同步退出状态给所有进程
        MPI_Bcast(&should_continue, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }
    
    // 最终时间统计
    if (rank == 0) {
        // 处理最后一批猜测（如果有）
        if (!q.guesses.empty()) {
            auto start_hash = system_clock::now();
            bit32 state[4];
            
            for (string pw : q.guesses)
            {
                MD5Hash(pw, state);
            }
            
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
            
            history += q.guesses.size();
            q.guesses.clear();
        }
        
        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
        
        cout << "Total guesses generated: " << history << endl;
        cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
        cout << "Hash time:" << time_hash << "seconds"<<endl;
        cout << "Train time:" << time_train <<"seconds"<<endl;
    }
    
    MPI_Finalize();
    return 0;
}
