#include "ThreadPool.h"

#include <unistd.h>

namespace obamadb {

  int setCoreAffinity(int core_id) {
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    DLOG_IF(WARNING, core_id < 0 || core_id >= num_cores) << "Attempted to bind thread to non-existant core: " << core_id;
    core_id = core_id % num_cores;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t current_thread = pthread_self();
    return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
  }

  void *WorkerLoop(void *worker_params) {
    ThreadMeta *meta = reinterpret_cast<ThreadMeta*>(worker_params);
    setCoreAffinity(meta->thread_id);
    int epoch = 0;
    while (true) {
      pthread_barrier_wait(meta->barrier1);
      if (meta->stop) {
        break;
      } else {
        //printf("thread %d @ epoch %d\n", meta->thread_id, epoch);
        meta->fn_execute_();
      }
      pthread_barrier_wait(meta->barrier2);

      epoch++;
    }

    // printf("thread %d stopping\n", meta->thread_id);
    return NULL;
  }

} // namespace obamadb
