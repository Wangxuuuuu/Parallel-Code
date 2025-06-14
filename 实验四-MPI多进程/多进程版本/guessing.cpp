#include "PCFG.h"
#include <mpi.h>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <iostream>

using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
// MPI并行化的Generate函数
// 定义最大猜测口令长度（根据实际需求调整）
#define MAX_GUESS_LENGTH 256

void PriorityQueue::Generate(PT pt)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 计算PT的概率
    CalProb(pt);

    // 对于只有一个segment的PT
    if (pt.content.size() == 1)
    {
        segment* a = nullptr;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        } else if (pt.content[0].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        // 错误检查
        if (a == nullptr || pt.max_indices.empty() || pt.max_indices[0] == 0) {
            if (rank == 0) {
                std::cerr << "Error: Could not find segment or no values available" << std::endl;
            }
            return;
        }

        int total_iterations = pt.max_indices[0];
        std::vector<std::string> local_guesses;

        // 计算每个进程的任务分配
        int chunk_size = total_iterations / size;
        int remainder = total_iterations % size;
        
        // 确定当前进程的任务范围
        int start_idx = rank * chunk_size + (rank < remainder ? rank : remainder);
        int end_idx = start_idx + chunk_size + (rank < remainder ? 1 : 0);
        end_idx = std::min(end_idx, static_cast<int>(a->ordered_values.size()));

        // 生成本地猜测
        for (int i = start_idx; i < end_idx; i++) {
            local_guesses.push_back(a->ordered_values[i]);
        }

        // 通信：收集所有结果到主进程
        if (size > 1) {
            MPI_GatherGuesses(local_guesses, rank);
        } else {
            // 单进程模式直接添加
            guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
            total_guesses += local_guesses.size();
        }
    }
    // 对于多个segment的PT
    else
    {
        // 构建前缀（除最后一个segment外的所有部分）
        std::string prefix;
        for (int seg_idx = 0; seg_idx < pt.content.size() - 1; seg_idx++) {
            if (pt.content[seg_idx].type == 1) {
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[pt.curr_indices[seg_idx]];
            } else if (pt.content[seg_idx].type == 2) {
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[pt.curr_indices[seg_idx]];
            } else if (pt.content[seg_idx].type == 3) {
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[pt.curr_indices[seg_idx]];
            }
        }

        // 处理最后一个segment
        segment* a = nullptr;
        int last_seg_idx = pt.content.size() - 1;
        if (pt.content[last_seg_idx].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[last_seg_idx])];
        } else if (pt.content[last_seg_idx].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[last_seg_idx])];
        } else if (pt.content[last_seg_idx].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[last_seg_idx])];
        }

        // 错误检查
        if (a == nullptr || pt.max_indices.empty() || last_seg_idx >= pt.max_indices.size()) {
            if (rank == 0) {
                std::cerr << "Error: Invalid segment or index range" << std::endl;
            }
            return;
        }

        int total_iterations = pt.max_indices[last_seg_idx];
        std::vector<std::string> local_guesses;

        // 计算每个进程的任务分配
        int chunk_size = total_iterations / size;
        int remainder = total_iterations % size;
        
        // 确定当前进程的任务范围
        int start_idx = rank * chunk_size + (rank < remainder ? rank : remainder);
        int end_idx = start_idx + chunk_size + (rank < remainder ? 1 : 0);
        end_idx = std::min(end_idx, static_cast<int>(a->ordered_values.size()));

        // 生成本地猜测
        for (int i = start_idx; i < end_idx; i++) {
            local_guesses.push_back(prefix + a->ordered_values[i]);
        }

        // 通信：收集所有结果到主进程
        if (size > 1) {
            MPI_GatherGuesses(local_guesses, rank);
        } else {
            // 单进程模式直接添加
            guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
            total_guesses += local_guesses.size();
        }
    }
}
// 辅助函数：收集所有进程的猜测结果到主进程
void PriorityQueue::MPI_GatherGuesses(const std::vector<std::string>& local_guesses, int rank)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 收集每个进程生成了多少猜测
    int local_count = local_guesses.size();
    std::vector<int> all_counts(size);
    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // 计算接收缓冲区总大小
        int total_guesses_to_recv = 0;
        for (int i = 1; i < size; i++) {
            total_guesses_to_recv += all_counts[i];
        }
        
        // 预留空间
        guesses.reserve(guesses.size() + local_count + total_guesses_to_recv);
        
        // 添加主进程自己的猜测
        guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
        total_guesses += local_count;
        
        // 接收其他进程的猜测
        for (int src = 1; src < size; src++) {
            if (all_counts[src] > 0) {
                // 接收字符串长度数组
                std::vector<int> str_lengths(all_counts[src]);
                MPI_Recv(str_lengths.data(), all_counts[src], MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // 计算总字符数
                int total_chars = 0;
                for (int len : str_lengths) total_chars += len;
                
                // 接收字符串数据
                std::vector<char> buffer(total_chars);
                MPI_Recv(buffer.data(), total_chars, MPI_CHAR, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // 从缓冲区提取字符串
                char* ptr = buffer.data();
                for (int i = 0; i < all_counts[src]; i++) {
                    std::string guess(ptr, ptr + str_lengths[i]);
                    guesses.push_back(guess);
                    ptr += str_lengths[i];
                }
                total_guesses += all_counts[src];
            }
        }
    } else {
        // 非主进程发送数据
        if (local_count > 0) {
            // 发送字符串长度数组
            std::vector<int> str_lengths;
            for (const auto& guess : local_guesses) {
                str_lengths.push_back(guess.size());
            }
            MPI_Send(str_lengths.data(), local_count, MPI_INT, 0, 0, MPI_COMM_WORLD);
            
            // 计算总字符数并发送字符串数据
            int total_chars = 0;
            for (int len : str_lengths) total_chars += len;
            
            std::vector<char> buffer;
            for (const auto& guess : local_guesses) {
                buffer.insert(buffer.end(), guess.begin(), guess.end());
            }
            MPI_Send(buffer.data(), total_chars, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        }
    }
}
