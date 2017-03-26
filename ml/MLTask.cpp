#include "ml/MLTask.h"

namespace obamadb {
  namespace ml {
    /**
     * Dot product
     */
    num_t dot(const dvector <num_t> &v1, num_t const *d2) {
      num_t sum = 0;
      num_t const *__restrict__ pv1 = v1.values_;
      num_t const *__restrict__ pv2 = d2;
      for (int i = 0; i < v1.size(); ++i) {
        sum += pv1[i] * pv2[i];
      }
      return sum;
    }

    void scale(dvector <num_t> &v1, num_t e) {
      int size = v1.size();
      for (int i = 0; i < size; i++) {
        v1.values_[i] *= e;
      }
    }

    void scale_and_add(dvector <num_t> &v1, dvector <num_t> &v2, num_t e) {
      int size = v1.size();
      for (int i = 0; i < size; i++) {
        v1.values_[i] += v2.values_[i] * e;
      }
    }


  }  // namespace ml

} // namespace obamadb
