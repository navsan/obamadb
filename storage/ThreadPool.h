#ifndef OBAMADB_THREADPOOL_H_
#define OBAMADB_THREADPOOL_H_

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

namespace obamadb {

  class Task {
    virtual void run() = 0;
  };

  class GradientTask : Task {
    void run() override;
  };

  class ErrorTask : Task {
    void run() override;

    double getError() {
      return last_error_;
    }

    double last_error_;
  };

  /**
   * Github user progschj's implementation
   */
  class ThreadPool {
  public:
    ThreadPool(size_t);

    template<class F, class... Args>
    auto enqueue(F &&f, Args &&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

    ~ThreadPool();

  private:
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers;
    // the task queue
    std::queue<std::function<void()> > tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
  };

  // add new work item to the pool
  template<class F, class... Args>
  inline auto ThreadPool::enqueue(F &&f, Args &&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()> >(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);

      // don't allow enqueueing after stopping the pool
      if (stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");

      tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
  }

} // namespace obamadb


#endif //OBAMADB_THREADPOOL_H_
