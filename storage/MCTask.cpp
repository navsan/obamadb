#include "MCTask.h"

namespace obamadb {

  void MCTask::execute(int threadId, void *svm_state) {
    (void) svm_state; // silence compiler warning.
    (void) threadId;
  }

}
