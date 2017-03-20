#ifndef OBAMADB_SVMTASK_H
#define OBAMADB_SVMTASK_H

#include "ml/MLTask.h"
#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/exvector.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

namespace obamadb {

  /**
   * Hyper-parameters shared between many SVM tasks/workers.
   */
  struct SVMHyperParams {
    SVMHyperParams(float mu,
              float step_size,
              float step_decay)
      : mu(mu),
        step_size(step_size),
        step_decay(step_decay),
        degrees() {}

    float mu;
    float step_size;
    float step_decay;
    std::vector<int> degrees;
  };

  class SVMTask : MLTask {
  public:
    SVMTask(DataView *dataView,
            fvector *sharedTheta,
            SVMHyperParams *sharedParams)
      : MLTask(dataView),
        shared_theta_(sharedTheta),
        shared_params_(sharedParams) {}

    MLAlgorithm getType() override {
      return MLAlgorithm::kSVM;
    }

    /**
     * Calculates and applies the gradient of the SVM.
     */
    void execute(int thread_id, void *ml_state) override;

    /**
     * The number of misclassified examples in a training block.
     * @param theta The model.
     * @param block The block.
     * @return Number misclassified.
     */
    static int numMisclassified(const fvector &theta, const SparseDataBlock<num_t> &block);

    /**
     * Gets the fraction of misclassified examples.
     * @param theta The trained weights.
     * @param block A sample of the data.
     * @return Fraction of misclassified examples.
     */
    static double fractionMisclassified(const fvector &theta, std::vector<SparseDataBlock<num_t> *> const &block);

    /**
     * Root mean squared error.
     * @param theta The trained weights.
     * @param blocks All the data.
     */
    static double rmsError(const fvector &theta, std::vector<SparseDataBlock<num_t> *> const &block);

    /**
    * @param theta
    * @param blocks
    * @return
    */
    static double rmsErrorLoss(const fvector &theta, std::vector<SparseDataBlock<num_t> *> const &blocks);

    fvector *shared_theta_;
    SVMHyperParams *shared_params_;

    DISABLE_COPY_AND_ASSIGN(SVMTask);
  };

  /**
   * Constructs the SVM to the parameters used in the HW! paper.
   * @return Caller-owned SVM params.
   */
  template <class T>
  SVMHyperParams *DefaultSVMHyperParams(
      std::vector<SparseDataBlock<T> *> &all_blocks) {
    SVMHyperParams *params = new SVMHyperParams(1, 0.1, 0.99);

    int dim = 0;
    // count the number of members of each column
    std::vector<int> &degrees = params->degrees;

    for (int k = 0; k < all_blocks.size(); ++k) {
      const SparseDataBlock<T> &block = *all_blocks[k];
      if (dim < block.getNumColumns()) {
        dim = block.getNumColumns();
        degrees.resize(dim);
      }

      svector<float_t> row;
      for (int i = 0; i < block.getNumRows(); i++) {
        block.getRowVector(i, &row);
        for (int j = 0; j < row.numElements(); j++) {
          degrees[row.index_[j]] += 1;
        }
      }
    }

    return params;
  };
  }  // namespace obamadb

#endif //OBAMADB_SVMTASK_H
