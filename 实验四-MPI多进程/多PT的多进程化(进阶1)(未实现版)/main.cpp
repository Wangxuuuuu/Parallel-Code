#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <signal.h>
#include <cstdlib>
using namespace std;
using namespace chrono;

// 全局变量用于信号处理
volatile bool should_exit = false;

void signal_handler(int signal) {
    cout << "\nReceived signal " << signal << ", preparing to exit..." << endl;
    should_exit = true;
}

int main()
{
    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "Initialization completed, starting conservative parallel PT processing..." << endl;
    
    int curr_num = 0;
    auto start = system_clock::now();
    int history = 0;
    
    // **保守的并行处理配置**
    int num_parallel_pts = 4;  
    int batch_count = 0;       
    int max_batches = 1000;    // 降低最大批次数
    
    while (!q.priority.empty() && batch_count < max_batches && !should_exit)
    {
        batch_count++;
        
        // 每20个批次输出一次状态（减少输出频率）
        if (batch_count % 20 == 1) {
            cout << "\n=== Batch " << batch_count << " ===" << endl;
            cout << "Priority queue size: " << q.priority.size() << endl;
            fflush(stdout);
        }
        
        // **更保守的策略：减少并行处理频率**
        if (q.priority.size() >= num_parallel_pts && batch_count % 100 == 1) {
            // 每100个批次使用一次并行处理
            cout << "Using parallel processing for " << num_parallel_pts << " PTs" << endl;
            fflush(stdout);
            q.PopMultiple(num_parallel_pts);
        } else {
            // 主要使用单PT处理
            q.PopNext();
        }
        
        // 检查是否应该退出
        if (should_exit) {
            cout << "Received exit signal, breaking main loop..." << endl;
            break;
        }
        
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 200000)  // 减少输出频率
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 目标数量检查
            int generate_n = 10000000;
            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                
                cout << "\n=== Target Reached - Final Results ===" << endl;
                cout << "Total batches processed: " << batch_count << endl;
                cout << "Total guesses generated: " << history + q.total_guesses << endl;
                cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                cout << "Hash time: " << time_hash << " seconds" << endl;
                cout << "Train time: " << time_train << " seconds" << endl;
                break;
            }
        }
        
        // 内存管理 - 减少阈值避免内存问题
        if (curr_num > 500000)  // 降低阈值
        {
            cout << "Processing MD5 hashes for " << curr_num << " guesses..." << endl;
            fflush(stdout);
            
            auto start_hash = system_clock::now();
            bit32 state[4];
            
            int hash_count = 0;
            for (const string& pw : q.guesses)
            {
                if (should_exit) break;  // 检查退出信号
                
                MD5Hash(pw, state);
                hash_count++;

                // 减少哈希进度输出频率
                if (hash_count % 200000 == 0) {
                    cout << "  Processed " << hash_count << " hashes..." << endl;
                    fflush(stdout);
                }
            }

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            cout << "Hash processing completed for batch " << batch_count << endl;
            cout << "Batch hash time: " << double(duration.count()) * microseconds::period::num / microseconds::period::den << " seconds" << endl;
            fflush(stdout);

            // 更新历史记录并清空内存
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
            
            cout << "Memory cleared. Total processed so far: " << history << endl;
            fflush(stdout);
        }
        
        // 每50个批次输出详细状态
        if (batch_count % 50 == 0) {
            cout << "\n=== Status Update (Batch " << batch_count << ") ===" << endl;
            cout << "Current priority queue size: " << q.priority.size() << endl;
            cout << "Current guesses in memory: " << q.guesses.size() << endl;
            cout << "Total processed guesses: " << history + q.total_guesses << endl;
            
            auto current_time = system_clock::now();
            auto elapsed = duration_cast<microseconds>(current_time - start);
            double elapsed_seconds = double(elapsed.count()) * microseconds::period::num / microseconds::period::den;
            cout << "Average time per batch: " << elapsed_seconds / batch_count << " seconds" << endl;
            fflush(stdout);
        }
        
        // 安全检查：避免无限循环
        if (batch_count >= max_batches) {
            cout << "\nReached maximum batch limit (" << max_batches << "), exiting..." << endl;
            break;
        }
    }
    
    cout << "\nMain loop completed. Batch count: " << batch_count << endl;
    cout << "Priority queue final size: " << q.priority.size() << endl;
    cout << "Exit reason: ";
    if (should_exit) cout << "Signal received" << endl;
    else if (batch_count >= max_batches) cout << "Batch limit reached" << endl;
    else if (q.priority.empty()) cout << "Queue empty" << endl;
    else cout << "Target reached" << endl;
    fflush(stdout);
    
    // 处理剩余的猜测
    if (!q.guesses.empty() && !should_exit) {
        cout << "\nProcessing final batch of " << q.guesses.size() << " guesses..." << endl;
        fflush(stdout);
        
        auto start_hash = system_clock::now();
        bit32 state[4];
        
        for (const string& pw : q.guesses) {
            if (should_exit) break;
            MD5Hash(pw, state);
        }
        
        auto end_hash = system_clock::now();
        auto duration = duration_cast<microseconds>(end_hash - start_hash);
        time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
        
        history += q.guesses.size();
    }
    
    cout << "\n=== Conservative Multi-Process PCFG Execution Completed ===" << endl;
    cout << "Total batches: " << batch_count << endl;
    cout << "Total guesses: " << history << endl;
    cout << "Parallel processing configuration: " << num_parallel_pts << " PTs per batch" << endl;
    cout << "Final training time: " << time_train << " seconds" << endl;
    cout << "Final guess time: " << time_guess - time_hash << " seconds" << endl;
    cout << "Final hash time: " << time_hash << " seconds" << endl;
    
    // **确保程序正常退出**
    cout << "Program finished successfully." << endl;
    fflush(stdout);
    
    return 0;
}
