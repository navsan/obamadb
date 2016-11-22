#ifndef OBAMADB_MLTASK_H
#define OBAMADB_MLTASK_H

#include <cmath>
#include <memory>
#include <mutex>

#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/exvector.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

namespace obamadb {
  namespace ml {
    /**
     * The number of misclassified examples in a training block.
     * @param theta The model.
     * @param block The block.
     * @return Number misclassified.
     */
    int numMisclassified(const fvector &theta, const SparseDataBlock<float_t> &block);

    /**
     * Gets the fraction of misclassified examples.
     * @param theta The trained weights.
     * @param block A sample of the data.
     * @return Fraction of misclassified examples.
     */
    float_t fractionMisclassified(const fvector &theta, std::vector<SparseDataBlock<float_t>*> const & block);

    /**
     * Root mean squared error.
     * @param theta The trained weights.
     * @param blocks All the data.
     */
    float_t rmsError(const fvector &theta, std::vector<SparseDataBlock<float_t>*> const & block);

    /**
     * TODO: this is really SVM loss.
     * @param theta
     * @param blocks
     * @return
     */
    float_t rmsErrorLoss(const fvector &theta, std::vector<SparseDataBlock<float_t> *> const &blocks);

    /**
     * @return The L2 distance between two vectors.
     */
    float_t L2Distance(const fvector &v1, const fvector &v2);

  } // end namespace ml

  class MLTask {
  public:
    /*
     * Takes ownership of the DataView.
     */
    MLTask(DataView *dataView)
      : data_view_(dataView) {}

    ~MLTask() {
      delete data_view_;
    };

    /**
     * This method will be executed by a threadpool.
     *
     * @param thread_id The ID of the thread which is executing the task.
     * @param ml_state  Some associated state relevent to the computation.
     */
    virtual void execute(int thread_id, void* ml_state) = 0;

  protected:
    DataView *data_view_;

    DISABLE_COPY_AND_ASSIGN(MLTask);
  };

  struct SVMParams {
    SVMParams(float_t mu,
              float_t step_size,
              float_t step_decay)
      : mu(mu),
        step_size(step_size),
        step_decay(step_decay),
        degrees(),
    next_work_idx_(0),
    unassigned_work_(),
    work_mutex_(){}

    float_t mu;
    float_t step_size;
    float_t step_decay;
    std::vector<int> degrees;

    DataView *getWork() {
      // do a quick check without locking:
      if (next_work_idx_ == -1) {
        return nullptr;
      }

      int idx = -1;
      work_mutex_.lock();
      // someone may have grabbed the last bit of work
      if (next_work_idx_ != -1) {
        idx = next_work_idx_;
        next_work_idx_--;
      }
      work_mutex_.unlock();
      if (idx != -1) {
        return unassigned_work_[idx].get();
      } else {
        return nullptr;
      }
    }

    void roundReset() {
      next_work_idx_ = unassigned_work_.size() - 1;
    }

    /**
     * Takes ownership of the dataview.
     * @param dataView
     */
    void addWork(DataView* dataView) {
      unassigned_work_.emplace_back(std::unique_ptr<DataView>(dataView));
    }

  private:
    int next_work_idx_;
    std::vector<std::unique_ptr<DataView>> unassigned_work_;
    std::mutex work_mutex_;

  };

  class SVMTask : MLTask {
  public:
    SVMTask(DataView *dataView,
            fvector *sharedTheta,
            SVMParams *sharedParams)
      : MLTask(dataView),
        shared_theta_(sharedTheta),
        shared_params_(sharedParams) { }

    /**
     * Calculates and applies the gradient of the SVM.
     */
    void execute(int thread_id, void* ml_state) override;

    void gradientOnView(DataView* dataView);

    fvector* shared_theta_;
    SVMParams* shared_params_;

    DISABLE_COPY_AND_ASSIGN(SVMTask);
  };

  /**
   * Constructs the SVM to the parameters used in the HW! paper.
   * @return Caller-owned SVM params.
   */
  template<class T>
  SVMParams* DefaultSVMParams(std::vector<SparseDataBlock<T>*>& all_blocks) {
    SVMParams * params = new SVMParams(1, 0.1, 0.99);
    int dim = 0;
    std::vector<int>& degrees = params->degrees;

    for (int k = 0; k < all_blocks.size(); ++k) {
      const SparseDataBlock<T>& block = *all_blocks[k];
      if (dim < block.getNumColumns()) {
        dim = block.getNumColumns();
        degrees.resize(block.getNumColumns());
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

}



#endif //OBAMADB_MLTASK_H
