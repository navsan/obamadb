#include "ThreadPool.h"

namespace obamadb {

  void *WorkerLoop(void *worker_params) {
    ThreadMeta *meta = reinterpret_cast<ThreadMeta*>(worker_params);
    threading::setCoreAffinity(meta->thread_id);
    int epoch = 0;
    while (true) {
      threading::barrier_wait(meta->barrier1);
      if (meta->stop) {
        break;
      } else {
        meta->fn_execute_(meta->thread_id, meta->state_);
      }
      threading::barrier_wait(meta->barrier2);
      epoch++;
    }
    return NULL;
  }


  int threading::numCores() {
#if APPLE
    return 4;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
  }

} // namespace obamadb
