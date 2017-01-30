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

  MCParams* DefaultMCParams(UnorderedMatrix const * training_matrix) {
    MCParams* params = new MCParams(-1, 0.05, 0.9);
    params->degrees_l.reset(new int[training_matrix->numRows()]);
    params->degrees_r.reset(new int[training_matrix->numColumns()]);
    double sum = 0;
    for (int i = 0; i < training_matrix->numElements(); i++) {
      MatrixEntry const & entry = training_matrix->get(i);
      params->degrees_l.get()[entry.row]++;
      params->degrees_r.get()[entry.column]++;
      sum += entry.value;
    }
    params->mean = sum / training_matrix->numElements();
    return params;
  }

  void MCTask::execute(int threadId, void *state) {
    int allocSize = examples_->numElements()/total_threads_;
    int start_index = allocSize * threadId;
    int end_index = std::min(allocSize * (threadId + 1), examples_->numElements());
    double mean = shared_params_->mean;
    double step_size = shared_params_->step_size;
    double mu = shared_params_->mu;
    int* degrees_l = shared_params_->degrees_l.get();
    int* degrees_r = shared_params_->degrees_r.get();

    dvector<num_t> lrow(0, nullptr);
    dvector<num_t> rrow(0, nullptr);
    dvector<num_t> lrow_temp;

//    std::vector<int> indirect(allocSize);
//    for (int i = start_index; i < end_index; i++) {
//      indirect[i] = i;
//    }
//    std::random_shuffle ( indirect.begin(), indirect.end() );

    for (int i = start_index; i < end_index; i++) {
      MatrixEntry const &entry = examples_->get(i);
      int row_index = entry.row;
      int col_index = entry.column;
      num_t value = entry.value;

      mat_l_->getRowVectorFast(row_index, &lrow);
      mat_r_->getRowVectorFast(col_index, &rrow);

      double err = ml::dot(lrow, rrow.values_) + mean - value;
      double e = -(step_size * err);

//      for (int j = 0; j < lrow.num_elements_; j++) {
//        if (lrow.values_[j] < -100 || lrow.values_[j] > 100) {
//          printf("diverging1\n");
//        }
//        if (rrow.values_[j] < -100 || rrow.values_[j] > 100) {
//          printf("diverging2\n");
//        }
//      }

      lrow_temp.copy(lrow);
      ml::scale(lrow_temp, (num_t) (1 - mu * step_size / ((double) degrees_l[row_index])));
      ml::scale_and_add(lrow_temp, rrow, e);

      ml::scale(rrow, (num_t) (1 - mu * step_size / ((double) degrees_r[col_index])));
      ml::scale_and_add(rrow, lrow, e);

      lrow.copy(lrow_temp);

//      for (int j = 0; j < lrow.num_elements_; j++) {
//        if (lrow.values_[j] < -100 || lrow.values_[j] > 100) {
//          printf("diverging1\n");
//        }
//        if (rrow.values_[j] < -100 || rrow.values_[j] > 100) {
//          printf("diverging2\n");
//        }
//      }
    }
    if (threadId == 0) {
      shared_params_->step_size *= shared_params_->step_decay;
    }
  }

  double MCTask::rmse(MCState const* state, UnorderedMatrix const * probe, double mean) {
    double sq_err = 0.0;
    dvector<num_t> lvec(0, nullptr);
    dvector<num_t> rvec(0, nullptr);
    for (int i = 0; i < probe->numElements(); i++) {
      MatrixEntry const & entry = probe->get(i);
      state->mat_l->getRowVectorFast(entry.row, &lvec);
      state->mat_r->getRowVectorFast(entry.column, &rvec);
      double loss = ml::dot(lvec, rvec.values_) + mean - entry.value;
      sq_err += loss * loss;
    }
    return sqrt(sq_err);
  }


}
