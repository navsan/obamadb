#ifndef OBAMADB_MLTASK_H
#define OBAMADB_MLTASK_H

#include <cmath>
#include <mutex>

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
    num_t dot(const dvector <num_t> &v1, num_t const *d2);

    /**
     * Sparse dot product
     */
    // num_t dot(const svector <num_t> &v1, num_t *d2);
    inline num_t dot(const svector <num_t> &v1, num_t *d2) {
      num_t sum = 0;
      num_t const *const __restrict__ pv1 = v1.values_;
      int const *const __restrict__ pvi1 = v1.index_;
      num_t const *const __restrict__ pv2 = d2;
      for (int i = 0; i < v1.numElements(); ++i) {
        sum += pv1[i] * pv2[pvi1[i]];
      }
      return sum;
    }


    void scale(dvector <num_t> &v1, num_t e);

    /**
     * Dense scale and add
     */
    void scale_and_add(dvector <num_t> &v1, dvector <num_t> &v2, num_t e);

    /**
     * Sparse scale and add. Only updates indices present in delta.
     * Applies delta to theta with a constant scaling parameter applied to delta.
     * @param theta Sparse vector of weights.
     * @param delta Sparse vector of changes.
     * @param e Scaling constant
     */
    inline void scale_and_add(num_t *theta, const svector<num_t> &delta, const num_t e) {
      num_t *const __restrict__ tptr = theta;
      num_t const *__restrict__ const vptr = delta.values_;
      int const *__restrict__ const iptr = delta.index_;
      for (int i = 0; i < delta.num_elements_; i++) {
        const int idx = iptr[i];
        tptr[idx] = tptr[idx] + (vptr[i] * e);
      }
    }

  }  // namespace ml

  enum class MLAlgorithm {
    kSVM,
    kMC
  };

  class MLTask {
  public:
    /*
     * Takes ownership of the DataView.
     */
    MLTask(DataView *dataView)
      : data_view_(dataView) {}

    ~MLTask() {
      if (data_view_ != nullptr) {
        delete data_view_;
      }
    };

    virtual MLAlgorithm getType() = 0;

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
}



#endif //OBAMADB_MLTASK_H
