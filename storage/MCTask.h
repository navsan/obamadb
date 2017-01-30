#ifndef OBAMADB_MCTASK_H
#define OBAMADB_MCTASK_H

#include "storage/DenseDataBlock.h"
#include "storage/exvector.h"
#include "storage/MLTask.h"
#include "storage/UnorderedMatrix.h"
#include "storage/Utils.h"

#include <memory>

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
        degrees_l(nullptr),
        degrees_r(nullptr),
        mean(0){}

    float mu;
    float step_size;
    float step_decay;
    std::unique_ptr<int> degrees_l;
    std::unique_ptr<int> degrees_r;
    double mean;
  };

  MCParams* DefaultMCParams(UnorderedMatrix const * training_matrix);

  /**
   * There is a single MC state per set of matrix completion tasks.
   * Contains factorization info.
   */
  struct MCState {
    MCState(UnorderedMatrix const * training_matrix, int rank)
      : rank(rank),
        mat_l(nullptr),
        mat_r(nullptr) {
      mat_l = new DenseDataBlock<num_t>(training_matrix->numRows(), rank);
      mat_r = new DenseDataBlock<num_t>(training_matrix->numColumns(), rank);
      mat_l->randomize();
      mat_r->randomize();
    }

    ~MCState() {
      delete mat_l;
      delete mat_r;
    }

    int rank;
    DenseDataBlock<num_t>* mat_l;
    DenseDataBlock<num_t>* mat_r;
  };

  class MCTask : MLTask {
  public:
    MCTask(int total_threads,
           UnorderedMatrix const * examples,
           MCState *sharedState,
           MCParams *sharedParams)
      : MLTask(nullptr),
        total_threads_(total_threads),
        examples_(examples),
        mat_l_(sharedState->mat_l),
        mat_r_(sharedState->mat_r),
        shared_params_(sharedParams) { }

    MLAlgorithm getType() override {
      return MLAlgorithm::kMC;
    }

    void execute(int thread_id, void* mc_state) override;

    static double rmse(MCState const* state, UnorderedMatrix const * probe, double mean);

    int total_threads_;
    UnorderedMatrix const * examples_;
    DenseDataBlock<num_t> *mat_l_;
    DenseDataBlock<num_t> *mat_r_;
    MCParams *shared_params_;

    DISABLE_COPY_AND_ASSIGN(MCTask);
  };

} // namespace obamadb

#endif //OBAMADB_MCTASK_H
