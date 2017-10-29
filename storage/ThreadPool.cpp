#include "storage/Utils.h"

#include "glog/logging.h"
#include <gflags/gflags.h>

#include "ThreadPool.h"

DECLARE_string(core_affinities);

namespace obamadb {

  // namespace threading {
  //   int getCoreAffinity() {
  //     if (FLAGS_core_affinities.compare("-1") == 0) {
  //       return NumThreadsAffinitized++;
  //     } else if (CoreAffinities.size() == 0) {
  //       std::vector<int> parsedAffinities = GetIntList(FLAGS_core_affinities);
  //       CoreAffinities.insert(CoreAffinities.begin(), parsedAffinities.begin(), parsedAffinities.end());
  //     }
  //     CHECK(CoreAffinities.size() > 0) << "invalid core_affinity flag";
  //     int assigned = CoreAffinities[NumThreadsAffinitized % CoreAffinities.size()];
  //     NumThreadsAffinitized++;
  //     return assigned;
  //   }
  // }

  void *WorkerLoop(void *worker_params) {
    ThreadMeta *meta = reinterpret_cast<ThreadMeta*>(worker_params);
    threading::setCoreAffinity(meta->core_id);
    int epoch = 0;
    while (true) {
      meta->barrier1->wait();
      if (meta->stop) {
        break;
      } else {
        meta->fn_execute_(meta->worker_id, meta->state_);
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
