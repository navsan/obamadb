#ifndef OBAMADB_THREADPOOL_H_
#define OBAMADB_THREADPOOL_H_

#ifdef __APPLE__
#define APPLE 1
#else
#define APPLE 0
#endif

#include <iostream>
#include <functional>
#include <printf.h>
#include <unistd.h>
#include <vector>
#ifndef APPLE
#include <pthread.h>
#endif

#include "storage/Task.h"

namespace obamadb {

  // Keep the threading mac-compadible.
  namespace threading {

#if APPLE
    typedef int pthread_barrierattr_t;
    typedef struct {
      pthread_mutex_t mutex;
      pthread_cond_t cond;
      int count;
      int tripCount;
    } pthread_barrier_t;
#endif

typedef pthread_barrier_t barrier_t;

    inline int barrier_init(pthread_barrier_t *barrier, const pthread_barrierattr_t *attr, unsigned int count) {
#if APPLE
      if (count == 0) {
        errno = EINVAL;
        return -1;
      }
      if (pthread_mutex_init(&barrier->mutex, 0) < 0) {
        return -1;
      }
      if (pthread_cond_init(&barrier->cond, 0) < 0) {
        pthread_mutex_destroy(&barrier->mutex);
        return -1;
      }
      barrier->tripCount = count;
      barrier->count = 0;
      return 0;
#else
      return pthread_barrier_init(barrier, attr, count);
#endif
    }

    inline int barrier_destroy(pthread_barrier_t *barrier) {
#if APPLE
      pthread_cond_destroy(&barrier->cond);
      pthread_mutex_destroy(&barrier->mutex);
      return 0;
#else
      return pthread_barrier_destroy(barrier);
#endif
    }

    inline int barrier_wait(pthread_barrier_t *barrier) {
#if APPLE
      pthread_mutex_lock(&barrier->mutex);
      ++(barrier->count);
      if (barrier->count >= barrier->tripCount) {
        barrier->count = 0;
        pthread_cond_broadcast(&barrier->cond);
        pthread_mutex_unlock(&barrier->mutex);
        return 1;
      } else {
        pthread_cond_wait(&barrier->cond, &(barrier->mutex));
        pthread_mutex_unlock(&barrier->mutex);
        return 0;
      }
#else
      return pthread_barrier_wait(barrier);
#endif
    }

   /**
    * Sets the current thread's core affinity. If the given core id is greater than the
    * number of cores present, then we chose the core which is specified core modulo num_cores
    * @param core_id Core to bind to.
    * @return Return code of the pthread_set_affinity call.
    */
    inline int setCoreAffinity(int core_id) {
#if APPLE
     // TODO: Is there no way to bind threads in apple?
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
  }

/*
 * Information relating to the running state of a thread.
 */
struct ThreadMeta {
  ThreadMeta(int thread_id,
             threading::barrier_t *barrier1,
             threading::barrier_t *barrier2,
             std::function<void()> task_fn) :
    thread_id(thread_id),
    barrier1(barrier1),
    barrier2(barrier2),
    fn_execute_(task_fn),
    stop(false) {}

  int thread_id;

  threading::barrier_t *barrier1;
  threading::barrier_t *barrier2;

  std::function<void()> fn_execute_;

  bool stop;
};

// takes the thread_t pointer as an argument.
void* WorkerLoop(void *worker_params);

/*
 * A simple static-task thread pool.
 */
class ThreadPool {
public:
  ThreadPool(std::vector<SVMTask*>& tasks)
    : meta_info_(), threads_(tasks.size()), num_workers_(tasks.size()) {
    threading::barrier_init(&b1_, NULL, num_workers_ + 1);
    threading::barrier_init(&b2_, NULL, num_workers_ + 1);

    for (int i = 0; i < tasks.size(); ++i) {
      meta_info_.push_back(ThreadMeta(i, &b1_, &b2_, std::bind(&SVMTask::execute, *tasks[i])));
    }
  }

  /**
   * Creates pthreads. Only call this method once.
   */
  void begin() {
    for (unsigned i = 0; i < num_workers_; i++) {
      pthread_create(&threads_[i], NULL, WorkerLoop, static_cast<void*>(&meta_info_[i]));
    }
  }

  void cycle() {
    threading::barrier_wait(&b1_);
    // workers do the routine
    threading::barrier_wait(&b2_);
    // workers are finished with routine and waiting on 1.
    // Here is an opportunity to re-allocate work, and do an update to the model.
  }

  void stop() {
    for (unsigned i = 0; i < num_workers_; i++) {
      meta_info_[i].stop = true;
    }
    threading::barrier_wait(&b1_);
    for (unsigned i = 0; i < num_workers_; i++) {
      pthread_join(threads_[i], NULL);
    }
  }

private:
  std::vector<ThreadMeta> meta_info_;
  std::vector<pthread_t> threads_;
  int num_workers_;

  threading::barrier_t b1_;
  threading::barrier_t b2_;
};

} // namespace obamadb


#endif //OBAMADB_THREADPOOL_H_
