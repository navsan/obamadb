#include "Task.h"

#include "storage/DataBlock.h"

namespace obamadb {

  double dot(const de_vector<double> &v1, double *d2) {
    double dotsum = 0;
    double * __restrict__ pv1 = v1.values_;
    double * __restrict__ pv2 = d2;
    for (int i = 0; i < v1.size(); ++i) {
      dotsum += pv1[i] * pv2[i];
    }
    return dotsum;
  }


  double dot(const se_vector<double> &v1, double *d2) {
    double dotsum = 0;
    double  * pv1 = v1.values_;
    int  * pvi1 = v1.index_;
    double  * pv2 = d2;
    for (int i = 0; i < v1.numElements(); ++i) {
      dotsum += pv1[i] * pv2[pvi1[i]];
    }
    return dotsum;
  }

  double Task::error(const DoubleVector &theta, const SparseDataBlock<double> &block) {
    // Percent of misclassified training examples.
    se_vector<double> row;
    double misclassed = 0;
    for (int i = 0; i < block.getNumRows(); i++) {
      block.getRowVector(i, &row);
      double sq_sum = dot(row, theta.values_);
      double classification = *row.getClassification();
      DCHECK(classification == 1 || classification == -1);

      if ((classification == 1 && sq_sum < 0) ||
        (classification == -1 && sq_sum >= 0)) {
        misclassed++;
      }
    }
    return misclassed / (double) block.getNumRows();
  }

  void SVMTask::execute() {
    data_view_->reset();
    se_vector<double> row;
    while (data_view_->getNext(&row)) {
      double const y = *row.getClassification();
      double wxy = dot(row, model_->values_);
      wxy = wxy * y; // {-1, 1}
      // hinge active
      if (wxy < 1) {
        double const e = params_.step_size * y;
        // scale weights
        double * theta = model_->values_;
        for (int i = 0; i < row.numElements(); i++) {
          theta[row.index_[i]] = theta[row.index_[i]] + (row.values_[i] * e);
        }
      }

      double const scalar = params_.step_size * params_.mu;
      // scale only the values which were updated.
      for (int i = row.numElements(); i-- > 0;) {
        const int idx_j = row.index_[i];
        double const deg = params_.degrees[idx_j];
        model_->values_[idx_j] *= 1 - scalar / deg;
      }
    }

    params_.step_size = params_.step_size * params_.step_decay;
  }
}  // namespace obamadb