#ifndef OBAMADB_MCTASK_H
#define OBAMADB_MCTASK_H

#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/exvector.h"
#include "storage/Matrix.h"
#include "storage/MLTask.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

namespace obamadb {

  /**
   * Matrix Completion parameters
   */
  struct MCParams {
    MCParams(float mu,
             float step_size,
             float step_decay)
      : mu(mu),
        step_size(step_size),
        step_decay(step_decay),
        degrees_l(),
        degrees_r(){}

    float mu;
    float step_size;
    float step_decay;
    std::vector<int> degrees_l;
    std::vector<int> degrees_r;
  };

  class MCTask : MLTask {
  public:
    MCTask(DataView *dataView,
           Matrix *lmatrix,
           Matrix *rmatrix,
           MCParams *sharedParams)
      : MLTask(dataView),
        mat_l(lmatrix),
        mat_r(rmatrix),
        shared_params_(sharedParams) { }

    MLAlgorithm getType() override {
      return MLAlgorithm::kMC;
    }

    void execute(int thread_id, void* ml_state) override;

    Matrix * mat_l;
    Matrix * mat_r;
    MCParams* shared_params_;

    DISABLE_COPY_AND_ASSIGN(MCTask);
  };

} // namespace obamadb

#endif //OBAMADB_MCTASK_H
