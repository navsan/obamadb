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
   * There is a single MC state per set of matrix completion tasks.
   * Contains factorization info.
   */
  struct MCState {
    MCState(UnorderedMatrix const * training_matrix, int rank)
      : mu(-1),
        step_size(0.05),
        step_decay(0.7),
        degrees_l(nullptr),
        degrees_r(nullptr),
        mean(0),
        rank(rank),
        mat_l(nullptr),
        mat_r(nullptr){
      mat_l = new DenseDataBlock<num_t>(training_matrix->numRows(), rank);
      mat_r = new DenseDataBlock<num_t>(training_matrix->numColumns(), rank);
      mat_l->randomize();
      mat_r->randomize();

      degrees_l.reset(new int[training_matrix->numRows()]);
      degrees_r.reset(new int[training_matrix->numColumns()]);
      double sum = 0;
      for (int i = 0; i < training_matrix->numElements(); i++) {
        MatrixEntry const & entry = training_matrix->get(i);
        degrees_l.get()[entry.row]++;
        degrees_r.get()[entry.column]++;
        sum += entry.value;
      }
      mean = sum / training_matrix->numElements();
    }

    ~MCState() {
      delete mat_l;
      delete mat_r;
    }

    float mu;
    float step_size;
    float step_decay;
    std::unique_ptr<int> degrees_l;
    std::unique_ptr<int> degrees_r;
    double mean;

    int rank;
    DenseDataBlock<num_t>* mat_l;
    DenseDataBlock<num_t>* mat_r;
  };

  class MCTask : MLTask {
  public:
    MCTask(int total_threads,
           UnorderedMatrix const * examples,
           MCState *sharedState)
      : MLTask(nullptr),
        total_threads_(total_threads),
        examples_(examples) { }

    MLAlgorithm getType() override {
      return MLAlgorithm::kMC;
    }

    void execute(int thread_id, void* mc_state) override;

    static double rmse(MCState const* state, UnorderedMatrix const * probe);

    int total_threads_;
    UnorderedMatrix const * examples_;
    MCState *shared_state_;

    DISABLE_COPY_AND_ASSIGN(MCTask);
  };

} // namespace obamadb

#endif //OBAMADB_MCTASK_H
