#include "storage/exvector.h"

#include <algorithm>
#include <vector>

#include "MCTask.h"


namespace obamadb {
  namespace ml {

    /**
     * Dot product
     */
    num_t dot(const dvector<num_t> &v1, num_t const *d2) {
      num_t sum = 0;
      num_t const *__restrict__ pv1 = v1.values_;
      num_t const *__restrict__ pv2 = d2;
      for (int i = 0; i < v1.size(); ++i) {
        sum += pv1[i] * pv2[i];
      }
      return sum;
    }

    void scale(dvector<num_t> & v1, num_t e) {
      int size = v1.size();
      for (int i = 0; i < size; i++) {
        v1.values_[i] *= e;
      }
    }

    void scale_and_add(dvector<num_t> & v1, dvector<num_t> & v2, num_t e) {
      int size = v1.size();
      for (int i = 0; i < size; i++) {
        v1.values_[i] += v2.values_[i] * e;
      }
    }
  }

  void MCTask::execute(int threadId, void *state) {
    int const allocSize = examples_->numElements()/total_threads_;
    int const start_index = allocSize * threadId;
    int const end_index = std::min(allocSize * (threadId + 1), examples_->numElements());
    double const mean = shared_state_->mean;
    double const step_size = shared_state_->step_size;
    double const mu = shared_state_->mu;
    int const * degrees_l = shared_state_->degrees_l.get();
    int const * degrees_r = shared_state_->degrees_r.get();
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
