#include "storage/MLTask.h"

#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/exvector.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

namespace obamadb {

  namespace ml {

    /**
     * Dot product
     */
    float_t dot(const dvector<float_t> &v1, float_t *d2) {
      float_t dotsum = 0;
      float_t *__restrict__ pv1 = v1.values_;
      float_t *__restrict__ pv2 = d2;
      for (int i = 0; i < v1.size(); ++i) {
        dotsum += pv1[i] * pv2[i];
      }
      return dotsum;
    }

    /**
     * Dot product
     */
    float_t dot(const svector<float_t> &v1, float_t *d2) {
      float_t dotsum = 0;
      float_t const *const __restrict__ pv1 = v1.values_;
      int const *const __restrict__ pvi1 = v1.index_;
      float_t const *const __restrict__ pv2 = d2;
      for (int i = 0; i < v1.numElements(); ++i) {
        dotsum += pv1[i] * pv2[pvi1[i]];
      }
      return dotsum;
    }

    int numMisclassified(const fvector &theta, const SparseDataBlock<float_t> &block) {
      svector<float_t> row(0, nullptr);
      int misclassified = 0;
      for (int i = 0; i < block.getNumRows(); i++) {
        block.getRowVectorFast(i, &row);
        const float_t dot_prod = dot(row, theta.values_);
        const float_t classification = *row.getClassification();
        DCHECK(classification == 1 || classification == -1) << "Expected binary classification.";

        misclassified+= (classification == 1 && dot_prod < 0) || (classification == -1 && dot_prod >= 0);
      }
      return misclassified;
    }

    float_t fractionMisclassified(const fvector &theta, std::vector<SparseDataBlock<float_t> *> const & blocks) {
      double total_misclassified = 0;
      double total_examples = 0;
      for (int i = 0; i < blocks.size(); i++) {
        SparseDataBlock<float_t> const *block = blocks[i];
        total_misclassified += numMisclassified(theta, *block);
        total_examples += block->getNumRows();
      }
      return (float_t) total_misclassified / total_examples;
    }

    float_t rmsError(const fvector &theta, std::vector<SparseDataBlock<float_t> *> const & blocks) {
      return (float_t) std::sqrt(fractionMisclassified(theta, blocks));
    }

    float_t rmsErrorLoss(const fvector &theta, std::vector<SparseDataBlock<float_t> *> const & blocks) {
      double total_examples = 0;
      float_t loss = 0;
      svector<float_t> row(0, nullptr);
      for (int i = 0; i < blocks.size(); i++) {
        SparseDataBlock<float_t> const &block = *blocks[i];

        for (int i = 0; i < block.getNumRows(); i++) {
          block.getRowVector(i, &row);
          const float_t dot_prod = dot(row, theta.values_);
          const float_t classification = *row.getClassification();
          DCHECK(classification == 1 || classification == -1);
          loss += std::max(1 - dot_prod * classification, static_cast<float_t >(0.0));
        }
        total_examples += block.getNumRows();
      }
      return (float_t) std::sqrt(loss) / std::sqrt(total_examples);
    }

    float_t L2Distance(const fvector &v1, const fvector &v2) {
      DCHECK_EQ(v1.dimension_, v2.dimension_);

      double dist_sum = 0;
      for (int i = 0; i < v1.dimension_; i++) {
        dist_sum += std::pow(v1.values_[i] - v2.values_[i], 2);
      }
      return (float_t) std::sqrt(dist_sum);
    }

    /**
     * Applies delta to theta with a constant scaling parameter applied to delta.
     * @param theta Sparse vector of weights.
     * @param delta Sparse vector of changes.
     * @param e Scaling constant
     */
    inline void scaleAndAdd(float_t* theta, const svector<float_t>& delta, const float_t e) {
      float_t * const __restrict__ tptr = theta;
      float_t const * __restrict__ const vptr = delta.values_;
      int const * __restrict__ const iptr = delta.index_;
      for (int i = 0; i < delta.num_elements_; i++) {
        const int idx = iptr[i];
        tptr[idx] = tptr[idx] + (vptr[i] * e);
      }
    }
  } // namespace ml

void SVMTask::execute(int threadId, void* svm_state) {
  (void) svm_state; // silence compiler warning.

  data_view_->reset();
  svector<float_t> row(0, nullptr); // a readonly se_vector.
  float_t * theta = shared_theta_->values_;
  const float_t mu = shared_params_->mu;
  const float_t step_size = shared_params_->step_size;

  // perform update with all the data in its view,
  while (data_view_->getNext(&row)) {
    float_t const y = *row.getClassification();
    float_t wxy = ml::dot(row, theta);
    wxy = wxy * y; // {-1, 1}
    // hinge active
    if (wxy < 1) {
      float_t const e = step_size * y;
      // scale weights
      ml::scaleAndAdd(theta, row, e);
    }

    float_t const scalar = step_size * mu;
    // scale only the values which were updated.
    for (int i = row.numElements(); i-- > 0;) {
      const int idx_j = row.index_[i];
      float_t const deg = shared_params_->degrees[idx_j];
      theta[idx_j] *= 1 - scalar / deg;
    }
  }

  if (threadId == 0) {
    shared_params_->step_size = step_size * shared_params_->step_decay;
  }

}

} // namespace obamadb