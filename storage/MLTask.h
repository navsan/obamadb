#ifndef OBAMADB_MLTASK_H
#define OBAMADB_MLTASK_H

#include <cmath>
#include <mutex>

#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/exvector.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

namespace obamadb {

  enum class MLAlgorithm {
    kSVM,
    kMC
  };

  class MLTask {
  public:
    /*
     * Takes ownership of the DataView.
     */
    MLTask(DataView *dataView)
      : data_view_(dataView) {}

    ~MLTask() {
      delete data_view_;
    };

    virtual MLAlgorithm getType() = 0;

    /**
     * This method will be executed by a threadpool.
     *
     * @param thread_id The ID of the thread which is executing the task.
     * @param ml_state  Some associated state relevent to the computation.
     */
    virtual void execute(int thread_id, void* ml_state) = 0;

  protected:
    DataView *data_view_;

    DISABLE_COPY_AND_ASSIGN(MLTask);
  };
}



#endif //OBAMADB_MLTASK_H
