#ifndef OBAMADB_CONVERGENCEOBSERVER_H_
#define OBAMADB_CONVERGENCEOBSERVER_H_

#include "ml/MLTask.h"
#include "ml/SVMTask.h"
#include "storage/ThreadPool.h"

#include <unistd.h>
#include <vector>

namespace obamadb {
/**
 * Used to snoop on the inter-epoch values of the model.
 */
class ConvergenceObserver {
public:
ConvergenceObserver(fvector const * const sharedTheta) :
model_ref_(sharedTheta),
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



} // namespace obamadb


#endif // #ifndef OBAMADB_CONVERGENCEOBSERVER_H_
