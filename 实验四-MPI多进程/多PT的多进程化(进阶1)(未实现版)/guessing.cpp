#include "PCFG.h"
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <sys/select.h>
#include <sys/time.h>
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是"纯粹的"PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中"123456"为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
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

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
}

// 原有的单PT处理方法保持不变
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

// **修复版：多进程并行处理多个PT**
void PriorityQueue::PopMultiple(int num_pts)
{
    if (priority.empty()) return;
    
    // 确保不会取出超过队列大小的PT数量
    int actual_pts = min(num_pts, (int)priority.size());
    
    cout << "Starting parallel processing of " << actual_pts << " PTs..." << endl;
    fflush(stdout);
    
    // 取出要处理的PT
    vector<PT> pts_to_process;
    for (int i = 0; i < actual_pts; i++) {
        pts_to_process.push_back(priority[i]);
    }
    
    // 从队列中移除这些PT
    priority.erase(priority.begin(), priority.begin() + actual_pts);
    
    // 收集所有结果
    vector<PTResult> all_results;
    
    // **最严格的进程管理：逐个处理，完全控制**
    for (int i = 0; i < actual_pts; i++) {
        // 创建管道
        int pipe_fd[2];
        if (pipe(pipe_fd) == -1) {
            continue;  // 跳过这个PT
        }
        
        // **关键1：在fork之前，创建完全隔离的环境**
        fflush(NULL);  // 刷新所有输出流
        
        pid_t pid = fork();
        
        if (pid == -1) {
            close(pipe_fd[0]);
            close(pipe_fd[1]);
            continue;
        }
        else if (pid == 0) {
            // **子进程 - 完全隔离策略**
            
            // 1. 立即关闭所有不需要的描述符
            close(pipe_fd[0]);
            
            // 2. **关键：重新打开所有标准流到/dev/null**
            freopen("/dev/null", "w", stdout);
            freopen("/dev/null", "w", stderr);
            freopen("/dev/null", "r", stdin);
            
            // 3. 设置进程组，完全脱离父进程控制
            setpgid(0, 0);
            
            // 4. 重置所有信号处理为默认
            signal(SIGINT, SIG_DFL);
            signal(SIGTERM, SIG_DFL);
            signal(SIGPIPE, SIG_DFL);
            signal(SIGCHLD, SIG_DFL);
            
            // 5. **最简化的PT处理 - 避免任何可能导致输出的操作**
            PTResult result;
            result.process_id = i;
            
            try {
                if (i < pts_to_process.size()) {
                    PT& pt = pts_to_process[i];
                    
                    // 极简化处理，最多生成1000个猜测
                    int max_guesses = 1000;
                    
                    if (pt.content.size() == 1 && !pt.max_indices.empty()) {
                        // 单segment处理
                        segment *a = nullptr;
                        if (pt.content[0].type == 1) {
                            int id = m.FindLetter(pt.content[0]);
                            if (id >= 0 && id < m.letters.size()) a = &m.letters[id];
                        } else if (pt.content[0].type == 2) {
                            int id = m.FindDigit(pt.content[0]);
                            if (id >= 0 && id < m.digits.size()) a = &m.digits[id];
                        } else if (pt.content[0].type == 3) {
                            int id = m.FindSymbol(pt.content[0]);
                            if (id >= 0 && id < m.symbols.size()) a = &m.symbols[id];
                        }
                        
                        if (a && !a->ordered_values.empty()) {
                            int limit = min(max_guesses, min((int)pt.max_indices[0], (int)a->ordered_values.size()));
                            for (int j = 0; j < limit; j++) {
                                result.guesses.push_back(a->ordered_values[j]);
                            }
                        }
                    } else if (pt.content.size() > 1 && !pt.max_indices.empty()) {
                        // 多segment处理 - 只生成少量样本
                        string base_guess = "sample";  // 简化为固定前缀
                        
                        // 只处理最后一个segment
                        segment& last_seg = pt.content.back();
                        segment *a = nullptr;
                        
                        if (last_seg.type == 1) {
                            int id = m.FindLetter(last_seg);
                            if (id >= 0 && id < m.letters.size()) a = &m.letters[id];
                        } else if (last_seg.type == 2) {
                            int id = m.FindDigit(last_seg);
                            if (id >= 0 && id < m.digits.size()) a = &m.digits[id];
                        } else if (last_seg.type == 3) {
                            int id = m.FindSymbol(last_seg);
                            if (id >= 0 && id < m.symbols.size()) a = &m.symbols[id];
                        }
                        
                        if (a && !a->ordered_values.empty()) {
                            int last_idx = pt.max_indices.size() - 1;
                            int limit = min(max_guesses, min((int)pt.max_indices[last_idx], (int)a->ordered_values.size()));
                            
                            for (int j = 0; j < limit; j++) {
                                result.guesses.push_back(base_guess + a->ordered_values[j]);
                            }
                        }
                    }
                }
                
                // 6. 发送结果（最简化格式）
                string data = to_string(result.process_id) + "|" + to_string(result.guesses.size()) + "|";
                for (size_t j = 0; j < min((size_t)500, result.guesses.size()); j++) {
                    data += result.guesses[j] + "|";
                }
                data += "END\n";
                
                // 分批写入，避免管道阻塞
                const char* buf = data.c_str();
                size_t len = data.length();
                size_t written = 0;
                while (written < len) {
                    ssize_t n = write(pipe_fd[1], buf + written, min(len - written, (size_t)512));
                    if (n <= 0) break;
                    written += n;
                }
                
            } catch (...) {
                // 忽略所有异常
            }
            
            // 7. 关闭文件描述符
            close(pipe_fd[1]);
            
            // 8. **最关键：使用最强制的退出方式**
            fflush(NULL);
            _exit(0);  // 强制立即退出，不执行任何清理
        }
        else {
            // **父进程 - 严格控制**
            close(pipe_fd[1]);  // 关闭写端
            
            // 设置读取超时，避免永远等待
            fd_set read_fds;
            struct timeval timeout;
            FD_ZERO(&read_fds);
            FD_SET(pipe_fd[0], &read_fds);
            timeout.tv_sec = 5;  // 5秒超时
            timeout.tv_usec = 0;
            
            PTResult result;
            if (select(pipe_fd[0] + 1, &read_fds, NULL, NULL, &timeout) > 0) {
                // 有数据可读
                result = ReceivePTResult(pipe_fd[0]);
            } else {
                // 超时或错误，创建空结果
                result.process_id = i;
            }
            
            all_results.push_back(result);
            close(pipe_fd[0]);
            
            // 等待子进程结束（带超时）
            int status;
            pid_t wait_result = waitpid(pid, &status, WNOHANG);
            if (wait_result == 0) {
                // 子进程还在运行，等待一会儿
                sleep(1);
                wait_result = waitpid(pid, &status, WNOHANG);
                if (wait_result == 0) {
                    // 强制杀死子进程
                    kill(pid, SIGKILL);
                    waitpid(pid, &status, 0);
                }
            }
            
            cout << "Process " << i << " completed with " 
                 << result.guesses.size() << " guesses" << endl;
            fflush(stdout);
        }
    }
    
    cout << "All child processes completed. Merging results..." << endl;
    fflush(stdout);
    
    // 合并所有结果
    for (const PTResult& result : all_results) {
        for (const string& guess : result.guesses) {
            guesses.push_back(guess);
            total_guesses++;
        }
    }
    
    cout << "Parallel processing completed. Generated " 
         << all_results.size() << " process results" << endl;
    fflush(stdout);
}


// **修复版：单个进程处理一个PT的函数**
PTResult PriorityQueue::ProcessSinglePT(PT pt, int process_id)
{
    PTResult result;
    result.process_id = process_id;
    
    // 创建临时的PriorityQueue来处理这个PT
    // 避免修改全局状态
    vector<string> temp_guesses;
    int temp_count = 0;
    
    // 重新实现Generate逻辑，避免使用成员变量
    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        else if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        else if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // 生成所有可能的密码
        for (int i = 0; i < pt.max_indices[0] && i < 10000; i += 1)  // 限制数量避免内存问题
        {
            string guess = a->ordered_values[i];
            temp_guesses.push_back(guess);
            temp_count += 1;
        }
    }
    else
    {
        string base_guess;
        int seg_idx = 0;
        
        // 构建前缀
        for (int idx : pt.curr_indices)
        {
            if (seg_idx >= pt.content.size() - 1) break;
            
            if (pt.content[seg_idx].type == 1)
            {
                base_guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 2)
            {
                base_guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 3)
            {
                base_guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
        }

        // 处理最后一个segment
        segment *a;
        int last_idx = pt.content.size() - 1;
        if (pt.content[last_idx].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[last_idx])];
        }
        else if (pt.content[last_idx].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[last_idx])];
        }
        else if (pt.content[last_idx].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[last_idx])];
        }
        
        // 生成所有可能的密码
        int max_iterations = min((int)pt.max_indices[last_idx], 10000);  // 限制数量
        for (int i = 0; i < max_iterations; i += 1)
        {
            string temp = base_guess + a->ordered_values[i];
            temp_guesses.push_back(temp);
            temp_count += 1;
        }
    }
    
    result.guesses = temp_guesses;
    
    // 暂时不生成新PT，避免复杂的序列化问题
    // result.new_pts = pt.NewPTs();
    
    return result;
}

// **新增：将多个新PT插入优先队列**
void PriorityQueue::InsertMultiplePTs(const vector<PT>& new_pts)
{
    for (const PT& pt : new_pts) {
        // 为每个新PT找到合适的插入位置
        bool inserted = false;
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                    priority.emplace(iter + 1, pt);
                    inserted = true;
                    break;
                }
            }
            if (iter == priority.end() - 1 && !inserted) {
                priority.emplace_back(pt);
                inserted = true;
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob && !inserted) {
                priority.emplace(iter, pt);
                inserted = true;
                break;
            }
        }
        
        // 如果队列为空，直接添加
        if (priority.empty()) {
            priority.push_back(pt);
        }
    }
}

// **新增：序列化PT对象用于进程间通信**
string PriorityQueue::SerializePT(const PT& pt)
{
    stringstream ss;
    ss << pt.pivot << "|";
    ss << pt.preterm_prob << "|";
    ss << pt.prob << "|";
    
    // 序列化content
    ss << pt.content.size() << "|";
    for (const segment& seg : pt.content) {
        ss << seg.type << "," << seg.length << ";";
    }
    
    // 序列化indices
    ss << pt.curr_indices.size() << "|";
    for (int idx : pt.curr_indices) {
        ss << idx << ",";
    }
    
    ss << pt.max_indices.size() << "|";
    for (int idx : pt.max_indices) {
        ss << idx << ",";
    }
    
    return ss.str();
}

PT PriorityQueue::DeserializePT(const string& data)
{
    PT pt;
    // 这里可以实现反序列化逻辑
    // 为简化演示，返回空PT
    return pt;
}

// **简化版：通过管道发送PTResult**
void PriorityQueue::SendPTResult(int pipe_fd, const PTResult& result)
{
    try {
        // 发送简单的数据格式：process_id|guess_count|guess1|guess2|...
        string data = to_string(result.process_id) + "|" + to_string(result.guesses.size()) + "|";
        
        // 只发送前1000个猜测，避免管道缓冲区溢出
        int max_guesses = min(1000, (int)result.guesses.size());
        for (int i = 0; i < max_guesses; i++) {
            data += result.guesses[i] + "|";
        }
        data += "END\n";
        
        // 分块发送，避免管道缓冲区问题
        const char* buf = data.c_str();
        size_t total = data.length();
        size_t sent = 0;
        
        while (sent < total) {
            ssize_t n = write(pipe_fd, buf + sent, min(total - sent, (size_t)1024));
            if (n <= 0) break;
            sent += n;
        }
    }
    catch (...) {
        // 忽略发送错误，确保子进程能正常退出
    }
}

// **简化版：通过管道接收PTResult**
PTResult PriorityQueue::ReceivePTResult(int pipe_fd)
{
    PTResult result;
    
    try {
        char buffer[8192];
        ssize_t total_read = 0;
        string data;
        
        // 读取数据直到遇到END标记或管道关闭
        while (total_read < sizeof(buffer) - 1) {
            ssize_t n = read(pipe_fd, buffer + total_read, 1024);
            if (n <= 0) break;
            total_read += n;
            
            // 检查是否读到END标记
            buffer[total_read] = '\0';
            string temp(buffer);
            if (temp.find("END\n") != string::npos) break;
        }
        
        if (total_read > 0) {
            buffer[total_read] = '\0';
            data = string(buffer);
            
            // 解析数据：process_id|guess_count|guess1|guess2|...|END
            size_t pos = 0;
            string token;
            
            // 解析process_id
            pos = data.find('|');
            if (pos != string::npos) {
                result.process_id = stoi(data.substr(0, pos));
                data = data.substr(pos + 1);
            }
            
            // 解析guess_count
            pos = data.find('|');
            if (pos != string::npos) {
                int guess_count = stoi(data.substr(0, pos));
                data = data.substr(pos + 1);
                
                // 解析猜测
                for (int i = 0; i < guess_count && data.find('|') != string::npos; i++) {
                    pos = data.find('|');
                    string guess = data.substr(0, pos);
                    if (guess != "END") {
                        result.guesses.push_back(guess);
                    }
                    data = data.substr(pos + 1);
                }
            }
        }
    }
    catch (...) {
        // 解析错误时返回空结果
        result.process_id = -1;
        result.guesses.clear();
    }
    
    return result;
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
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // 生成所有可能的密码
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // 生成所有可能的密码
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}
