#ifndef OBAMADB_CONVERGENCEOBSERVER_H_
#define OBAMADB_CONVERGENCEOBSERVER_H_

#include "ml/MLTask.h"
#include "ml/SVMTask.h"
#include "storage/ThreadPool.h"

#include <unistd.h>
#include <vector>

namespace obamadb {
/**
 * The function which a thread will call once per epoch to record the
 * model's state change over time.
 * @param tid Thread id. Not used.
 * @param observerState Pair<remaining epochs to observe, observer>.
 */
void observerThreadFn(int tid, void* observerState);

/**
 * Used to snoop on the inter-epoch values of the model.
 */
class ConvergenceObserver {
 public:
  ConvergenceObserver(fvector const* const sharedTheta)
      : model_ref_(sharedTheta),
        observedTimes_(),
        observedModels_(),
        cyclesObserved_(0),
        threadPool_(nullptr) {}

  void record() {
    auto current_time = std::chrono::steady_clock::now();
    observedTimes_.push_back(current_time.time_since_epoch().count());
    observedModels_.push_back(*model_ref_);
  }

  int size() {
    CHECK_EQ(observedModels_.size(), observedTimes_.size());
    return observedModels_.size();
  }

  static int kObserverWaitTimeUS;  // Time between observation captures.

  fvector const* const model_ref_;
  std::vector<std::uint64_t> observedTimes_;
  std::vector<fvector> observedModels_;
  int cyclesObserved_;
  ThreadPool* threadPool_;
};

int ConvergenceObserver::kObserverWaitTimeUS = 1000;

void observerThreadFn(int tid, void* observerState) {
  (void) tid;
  ConvergenceObserver* observer =
      reinterpret_cast<ConvergenceObserver*>(observerState);

  if (observer->cyclesObserved_ > 0) {
    return;
  }

  // Loop until the number of waiters is the number of worker threads.
  int const total_workers = observer->threadPool_->getNumWorkers();
  while (total_workers > observer->threadPool_->getWaiterCount()) {
    observer->record();
    usleep(ConvergenceObserver::kObserverWaitTimeUS);
  }
  observer->cyclesObserved_++;
}

} // namespace obamadb


#endif // #ifndef OBAMADB_CONVERGENCEOBSERVER_H_
