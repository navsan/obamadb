#include "Task.h"

#include "storage/DataBlock.h"

namespace obamadb {

  float_t dot(const de_vector<float_t> &v1, float_t *d2) {
    float_t dotsum = 0;
    float_t * __restrict__ pv1 = v1.values_;
    float_t * __restrict__ pv2 = d2;
    for (int i = 0; i < v1.size(); ++i) {
      dotsum += pv1[i] * pv2[i];
    }
    return dotsum;
  }


  float_t dot(const se_vector<float_t> &v1, float_t *d2) {
    float_t dotsum = 0;
    float_t const * const __restrict__ pv1 = v1.values_;
    int const * const __restrict__ pvi1  = v1.index_;
    float_t const * const __restrict__ pv2 = d2;
    for (int i = 0; i < v1.numElements(); ++i) {
      dotsum += pv1[i] * pv2[pvi1[i]];
    }
    return dotsum;
  }

  int Task::misclassified(const f_vector &theta, const SparseDataBlock<float_t> &block) {
    se_vector<float_t> row;
    int misclassed = 0;
    for (int i = 0; i < block.getNumRows(); i++) {
      block.getRowVector(i, &row);
      float_t sq_sum = dot(row, theta.values_);
      float_t classification = *row.getClassification();
      DCHECK(classification == 1 || classification == -1);

      if ((classification == 1 && sq_sum < 0) ||
          (classification == -1 && sq_sum >= 0)) {
        misclassed++;
      }
    }
    return misclassed;
  }

  float_t Task::error(const f_vector &theta, const SparseDataBlock<float_t> &block) {
    // Percent of misclassified training examples.
    return  (float_t)misclassified(theta, block) / (float_t) block.getNumRows();
  }

  float_t Task::error(const f_vector &theta, std::vector<SparseDataBlock<float_t> *> &blocks) {
    int total_misclassified = 0;
    int total_examples = 0;
    for (int i = 0; i < blocks.size(); i++) {
      SparseDataBlock<float_t> const * block = blocks[i];
      total_misclassified += misclassified(theta, *block);
      total_examples += block->getNumRows();
    }
    return (float_t) total_misclassified/ (float_t) total_examples;
  }

  inline void scaleAndAdd(float_t* theta, const se_vector<float_t>& vec, const float_t e) {
    float_t * const __restrict__ tptr = theta;
    float_t const * __restrict__ const vptr = vec.values_;
    int const * __restrict__ const iptr = vec.index_;
    for (int i = 0; i < vec.num_elements_; i++) {
      const int idx = iptr[i];
      theta[idx] = theta[idx] + (vptr[i] * e);
    }
  }

  void SVMTask::execute() {
    data_view_->reset();
    se_vector<float_t> row(0, nullptr); // a readonly se_vector.
    float_t * theta = model_->values_;
    while (data_view_->getNext(&row)) {
      float_t const y = *row.getClassification();
      float_t wxy = dot(row, model_->values_);
      wxy = wxy * y; // {-1, 1}
      // hinge active
      if (wxy < 1) {
        float_t const e = params_.step_size * y;
        // scale weights
        scaleAndAdd(theta, row, e);
      }

      float_t const scalar = params_.step_size * params_.mu;
      // scale only the values which were updated.
      for (int i = row.numElements(); i-- > 0;) {
        const int idx_j = row.index_[i];
        float_t const deg = params_.degrees[idx_j];
        theta[idx_j] *= 1 - scalar / deg;
      }
    }

    params_.step_size = params_.step_size * params_.step_decay;
  }
}  // namespace obamadb