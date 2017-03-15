#include "storage/exvector.h"

#include <algorithm>
#include <vector>

#include "MCTask.h"


namespace obamadb {

  void MCTask::execute(int threadId, void *state) {
    int const allocSize = examples_->numElements()/total_threads_;
    int const start_index = allocSize * threadId;
    int const end_index = std::min(allocSize * (threadId + 1), examples_->numElements());
    double const mean = shared_state_->mean;
    double const step_size = shared_state_->step_size;
    double const mu = shared_state_->mu;
    std::vector<int> const & degrees_l = shared_state_->degrees_l;
    std::vector<int> const & degrees_r = shared_state_->degrees_r;
    DenseDataBlock<num_t>* mat_l = shared_state_->mat_l.get();
    DenseDataBlock<num_t>* mat_r = shared_state_->mat_r.get();

    dvector<num_t> lrow(0, nullptr);
    dvector<num_t> rrow(0, nullptr);
    dvector<num_t> lrow_temp;

    for (int i = start_index; i < end_index; i++) {
      MatrixEntry const &entry = examples_->get(i);
      int row_index = entry.row;
      int col_index = entry.column;
      num_t value = entry.value;

      mat_l->getRowVectorFast(row_index, &lrow);
      mat_r->getRowVectorFast(col_index, &rrow);

      double err = ml::dot(lrow, rrow.values_) + mean - value;
      double e = -(step_size * err);

      lrow_temp.copy(lrow);
      ml::scale(lrow_temp, (num_t) (1 - mu * step_size / ((double) degrees_l[row_index])));
      ml::scale_and_add(lrow_temp, rrow, e);

      ml::scale(rrow, (num_t) (1 - mu * step_size / ((double) degrees_r[col_index])));
      ml::scale_and_add(rrow, lrow, e);

      lrow.copy(lrow_temp);
    }
    if (threadId == 0) {
      shared_state_->step_size *= shared_state_->step_decay;
    }
  }

  double MCTask::rmse(MCState const* state, UnorderedMatrix const * probe) {
    double sq_err = 0.0;
    dvector<num_t> lvec(0, nullptr);
    dvector<num_t> rvec(0, nullptr);
    for (int i = 0; i < probe->numElements(); i++) {
      MatrixEntry const & entry = probe->get(i);
      state->mat_l->getRowVectorFast(entry.row, &lvec);
      state->mat_r->getRowVectorFast(entry.column, &rvec);
      double loss = ml::dot(lvec, rvec.values_) + state->mean - entry.value;
      sq_err += loss * loss;
    }
    return sqrt(sq_err/probe->numElements());
  }


}
