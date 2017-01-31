#include "ThreadPool.h"

namespace obamadb {

  namespace threading {
    int getCoreAffinity() {
      if (FLAGS_core_affinities.compare("-1") == 0) {
        return NumThreadsAffinitized++;
      } else if (CoreAffinities.size() == 0) {
        char const * raw = FLAGS_core_affinities.c_str();
        int core = 0;
        for (int i = 0; i < FLAGS_core_affinities.size(); i++) {
          if (raw[i] == ',') {
            CoreAffinities.push_back(core);
            core = 0;
          } else {
            CHECK_LT(raw[i], 48 + 10) << "invalid core_affinity";
            CHECK_GE(raw[i], 48) << "invalid core_affinity";
            core *= 10;
            core += raw[i] - 48;
          }
        }
        CoreAffinities.push_back(core);
      }
      CHECK(CoreAffinities.size() > 0) << "invalid core_affinity flag";
      int assigned = CoreAffinities[NumThreadsAffinitized % CoreAffinities.size()];
      NumThreadsAffinitized++;
      return assigned;
    }
  }

  void *WorkerLoop(void *worker_params) {
    ThreadMeta *meta = reinterpret_cast<ThreadMeta*>(worker_params);
    int assigned_core = threading::getCoreAffinity();
    threading::setCoreAffinity(assigned_core);
    int epoch = 0;
    while (true) {
      meta->barrier1->wait();
      if (meta->stop) {
        break;
      } else {
        meta->fn_execute_(meta->thread_id, meta->state_);
      }
      meta->barrier2->wait();
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
