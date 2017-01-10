#include <cmath>

#include "storage/StorageConstants.h"

#include "storage/Utils.h"

namespace obamadb {

  fvector fvector::GetRandomFVector(int const dim) {
    fvector shared_theta(dim);
    // initialize to values [-1,1]
    for (unsigned i = 0; i < dim; ++i) {
      shared_theta[i] = static_cast<float_t>((1.0 - fmod((double) rand() / 100.0, 2)) / 10.0);
    }
    return shared_theta;
  }

}