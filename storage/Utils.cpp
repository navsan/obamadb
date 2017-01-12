#include <cmath>

#include "storage/StorageConstants.h"

#include "storage/Utils.h"

namespace obamadb {

  fvector fvector::GetRandomFVector(int const dim) {
    fvector shared_theta(dim);
    // initialize to values [-1000000,1000000]
    for (unsigned i = 0; i < dim; ++i) {
      //shared_theta[i] = (rand() % kScaleTheta * 2) - (kScaleTheta);
      shared_theta[i] = 0;
    }
    return shared_theta;
  }

}