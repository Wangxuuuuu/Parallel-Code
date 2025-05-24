#pragma once
#include <vector>
#include <queue>
#include <functional>
#include <atomic>
#include <pthread.h>

class ThreadPool {
public:
    ThreadPool(size_t);
    ~ThreadPool();

    // 提交任务
    template<class F>
    void enqueue(F&& f);

    void wait_all();

private:
    std::vector<pthread_t> workers;  // 使用pthread_t替代std::thread
    std::queue<std::function<void()>> tasks;

    pthread_mutex_t queue_mutex;  // 使用pthread_mutex_t
    pthread_cond_t condition;     // 使用pthread_cond_t
    std::atomic<bool> stop;
    std::atomic<int> working_count;
    
    static void* worker_thread(void* arg);  // 线程工作函数
};

inline ThreadPool::ThreadPool(size_t threads) : stop(false), working_count(0) {
    // 初始化互斥量和条件变量
    pthread_mutex_init(&queue_mutex, nullptr);
    pthread_cond_init(&condition, nullptr);

    for (size_t i = 0; i < threads; ++i) {
        pthread_t thread;
        pthread_create(&thread, nullptr, worker_thread, this);  // 创建线程
        workers.push_back(thread);
    }
}

template<class F>
void ThreadPool::enqueue(F&& f) {
    {
        pthread_mutex_lock(&queue_mutex);  // 锁定队列
        tasks.emplace(std::forward<F>(f));
    }
    pthread_cond_signal(&condition);  // 通知一个线程有任务可做
}

void ThreadPool::wait_all() {
    pthread_mutex_lock(&queue_mutex);  // 锁定队列
    pthread_cond_wait(&condition, &queue_mutex);  // 等待所有任务完成
    pthread_mutex_unlock(&queue_mutex);
}

void* ThreadPool::worker_thread(void* arg) {
    ThreadPool* pool = static_cast<ThreadPool*>(arg);
    while (true) {
        std::function<void()> task;
        {
            pthread_mutex_lock(&pool->queue_mutex);  // 锁定队列
            pthread_cond_wait(&pool->condition, &pool->queue_mutex);  // 等待任务
            if (pool->stop && pool->tasks.empty()) {
                pthread_mutex_unlock(&pool->queue_mutex);
                break;
            }

            task = std::move(pool->tasks.front());
            pool->tasks.pop();
            ++pool->working_count;  // 工作线程数增加
        }

        task();  // 执行任务

        --pool->working_count;  // 工作线程数减少
        pthread_cond_signal(&pool->condition);  // 通知其他线程
    }
    return nullptr;
}

inline ThreadPool::~ThreadPool() {
    {
        pthread_mutex_lock(&queue_mutex);
        stop = true;
    }
    pthread_cond_broadcast(&condition);  // 通知所有线程退出

    for (pthread_t &worker : workers) {
        pthread_join(worker, nullptr);  // 等待线程结束
    }

    pthread_mutex_destroy(&queue_mutex);  // 销毁互斥量
    pthread_cond_destroy(&condition);    // 销毁条件变量
}
