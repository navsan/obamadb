#ifndef OBAMADB_THREADPOOL_H_
#define OBAMADB_THREADPOOL_H_

#include <iostream>
#include <pthread.h>
#include <functional>
#include <vector>
#include <printf.h>

#include "storage/Task.h"

namespace obamadb {

typedef pthread_barrier_t barrier_t;

/*
 * Information relating to the running state of a thread.
 */
struct ThreadMeta {
  ThreadMeta(int thread_id,
             barrier_t *barrier1,
             barrier_t *barrier2,
             std::function<void()> task_fn) :
    thread_id(thread_id),
    barrier1(barrier1),
    barrier2(barrier2),
    fn_execute_(task_fn),
    stop(false) {}

  int thread_id;

  barrier_t *barrier1;
  barrier_t *barrier2;

  std::function<void()> fn_execute_;

  bool stop;
};

// takes the thread_t pointer as an argument.
void* WorkerLoop(void *worker_params) {
  ThreadMeta *meta = reinterpret_cast<ThreadMeta*>(worker_params);
  int epoch = 0;
  while (true) {
    pthread_barrier_wait(meta->barrier1);
    if (meta->stop) {
      break;
    } else {
      // printf("thread %d @ epoch %d\n", meta->thread_id, epoch);
      meta->fn_execute_();
    }
    pthread_barrier_wait(meta->barrier2);

    epoch++;
  }

  // printf("thread %d stopping\n", meta->thread_id);
  return NULL;
}

/*
 * A simple static-task thread pool.
 */
class ThreadPool {
public:
  ThreadPool(std::vector<Task*>& tasks)
  //ThreadPool(std::vector<LinearRegressionTask*>& tasks)
    : meta_info_(), threads_(tasks.size()), num_workers_(tasks.size()) {
    pthread_barrier_init(&b1_, NULL, num_workers_ + 1);
    pthread_barrier_init(&b2_, NULL, num_workers_ + 1);

    for (int i = 0; i < tasks.size(); ++i) {
      meta_info_.push_back(ThreadMeta(i, &b1_, &b2_, std::bind(&Task::execute, *tasks[i])));
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
    pthread_barrier_wait(&b1_);
    // workers do the routine
    pthread_barrier_wait(&b2_);
    // workers are finished with routine and waiting on 1.
    // Here is an opportunity to re-allocate work, and do an update to the model.
  }

  void stop() {
    for (unsigned i = 0; i < num_workers_; i++) {
      meta_info_[i].stop = true;
    }
    pthread_barrier_wait(&b1_);
    for (unsigned i = 0; i < num_workers_; i++) {
      pthread_join(threads_[i], NULL);
    }
  }

private:
  std::vector<ThreadMeta> meta_info_;
  std::vector<pthread_t> threads_;
  int num_workers_;
  barrier_t b1_;
  barrier_t b2_;

};

} // namespace obamadb


#endif //OBAMADB_THREADPOOL_H_
