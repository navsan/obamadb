#ifndef OBAMADB_THREADPOOL_H_
#define OBAMADB_THREADPOOL_H_

#ifdef __APPLE__
#define APPLE 1
#else
#define APPLE 0
#endif

#include <condition_variable>
#include <functional>
#include <iostream>
#include <printf.h>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

#include "glog/logging.h"
#include <gflags/gflags.h>

DECLARE_string(core_affinities);

namespace obamadb {

  // Keep the threading mac-compadible.
  namespace threading {

    class barrier_t {
    public:
      barrier_t(int totalWaiters)
        : mutex_(),
          cond_(),
          count_(totalWaiters),
          epoch_(0),
          threshold_(totalWaiters) {}

      void wait() {
        int epoch_stackvar = epoch_;
        std::unique_lock<std::mutex> lock{mutex_};
        if (!--count_) {
          epoch_++;
          count_ = threshold_;
          cond_.notify_all();
        } else {
          cond_.wait(lock, [this, epoch_stackvar] { return epoch_stackvar != epoch_; });
        }
      }

      int count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
      }

    private:
      std::mutex mutex_;
      std::condition_variable cond_;
      int count_;
      int epoch_; // # times broken.
      int const threshold_;
    };

   /**
    * Sets the current thread's core affinity. If the given core id is greater than the
    * number of cores present, then we chose the core which is specified core modulo num_cores
    * @param core_id Core to bind to.
    * @return Return code of the pthread_set_affinity call.
    */
    inline int setCoreAffinity(int core_id) {
#if APPLE
     // Not part of the iOS interface.
     return 0;
#else
     int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
     DLOG_IF(WARNING, core_id < 0 || core_id >= num_cores) << "Attempted to bind thread to non-existant core: " << core_id;
     core_id = core_id % num_cores;

     cpu_set_t cpuset;
     CPU_ZERO(&cpuset);
     CPU_SET(core_id, &cpuset);
     pthread_t current_thread = pthread_self();
     return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
#endif
   }

    static int NumThreadsAffinitized;
    static std::vector<int> CoreAffinities;

    /**
     * Choose the optimal core. This is experimental. In the future we should get this automatically from
     * lib numa like in
     * https://github.com/apache/incubator-quickstep/pull/126/commits/248cec27341fa8f19658d0705c1dca3fec0ff550
     * @return The core to bind to.
     */
    int getCoreAffinity();

    int numCores();

  } // end namespace threading

/*
 * Information relating to the running state of a thread.
 */
struct ThreadMeta {
  ThreadMeta(int thread_id,
             threading::barrier_t *barrier1,
             threading::barrier_t *barrier2,
             std::function<void(int, void*)> task_fn,
             void* state) :
    thread_id(thread_id),
    barrier1(barrier1),
    barrier2(barrier2),
    fn_execute_(task_fn),
    state_(state),
    stop(false) {}

  int thread_id;

  threading::barrier_t *barrier1;
  threading::barrier_t *barrier2;

  std::function<void(int, void*)> fn_execute_;
  void* state_;

  bool stop;
};

// takes the thread_t pointer as an argument.
void* WorkerLoop(void *worker_params);

/*
 * A simple static-task thread pool.
 */
class ThreadPool {
public:
  ThreadPool(std::vector<std::function<void(int, void*)>> const & thread_fns,
             const std::vector<void*> &thread_states)
    : meta_info_(),
      threads_(),
      num_workers_(thread_states.size()),
      b1_(new threading::barrier_t(num_workers_ + 1)),
      b2_(new threading::barrier_t(num_workers_ + 1))
  {
    for (int i = 0; i < thread_states.size(); ++i) {
      meta_info_.push_back(ThreadMeta(i, b1_, b2_, thread_fns[i], thread_states[i]));
    }
  }

  /**
   * ctor
   *
   * Every thread will be given the same state. Useful in situations
   * where the thread_id passed to thread_fn determines the allotment of
   * data amonst the threads.
   *
   * @param thread_fn Function which thread will execute. The int param will the the thread id
   *      and the void* param will the the thread's thread local state.
   * @param shared_thread_state State to be shared amongst all threads
   */
  ThreadPool(std::function<void(int, void*)> thread_fn,
             void* shared_thread_state,
             int num_threads)
    : meta_info_(),
      threads_(),
      num_workers_(num_threads),
      b1_(new threading::barrier_t(num_workers_ + 1)),
      b2_(new threading::barrier_t(num_workers_ + 1))
  {
    for (int i = 0; i < num_workers_; ++i) {
      meta_info_.push_back(ThreadMeta(i, b1_, b2_, thread_fn, shared_thread_state));
    }
  }

  ~ThreadPool() {
    delete b1_;
    delete b2_;
    for (int i = 0; i < threads_.size(); i++){
      delete threads_[i];
    }
  }

  /**
   * Only call this method once.
   */
  void begin() {
    for (unsigned i = 0; i < num_workers_; i++) {
      threads_.push_back(new std::thread(WorkerLoop, static_cast<void*>(&meta_info_[i])));
    }
  }

  void cycle() {
    b1_->wait();
    // workers do the routine
    b2_->wait();
    // workers are finished with routine and waiting on 1.
    // Here is an opportunity to re-allocate work, and do an update to the model.
  }

  int getWaiterCount() const {
    return b2_->count();
  }

  int getNumWorkers() const {
    return num_workers_;
  }

  void stop() {
    for (unsigned i = 0; i < num_workers_; i++) {
      meta_info_[i].stop = true;
    }
    b1_->wait();
    for (unsigned i = 0; i < threads_.size(); i++) {
      threads_[i]->join();
    }
  }

private:
  std::vector<ThreadMeta> meta_info_;
  std::vector<std::thread*> threads_;
  int num_workers_;

  threading::barrier_t *b1_;
  threading::barrier_t *b2_;
};

} // namespace obamadb


#endif //OBAMADB_THREADPOOL_H_
