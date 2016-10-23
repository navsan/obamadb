#include "Task.h"

namespace obamadb {

  double Task::error(const DoubleVector &theta, const DataBlock &block) {
    // Percent of misclassified training examples.
    const unsigned N = block.getNumRows();
    const unsigned n_theta = theta.dimension_;
    double *cursor = block.getStore();
    long misclassed = 0;
    for (unsigned i = 0; i < N; ++i) {
      double a_theta = 0;
      for (unsigned j = 0; j < n_theta; ++j) {
        a_theta += theta[j] * *cursor;
        cursor += 1;
      }
      double classification = a_theta > 0.0 ? 1 : -1;
      double y = *cursor;

      DCHECK(y == -1.0 || y == 1.0);

      cursor++;

      misclassed += y != classification ? 1 : 0;
    }
    return misclassed / N;
  }

  SVMParams DefaultSVMParams(const std::vector<DataBlock *>& all_blocks) {
    int const dim = all_blocks[0]->getWidth() - 1;  // the assumption here is that the last element is that classification
    std::vector<int> degrees(dim);
    for(auto block : all_blocks) {
      for (int i = 0; i < block->getNumRows(); i++) {
        for (int j = 0; j < dim; j++) {
          degrees[j] += (block->get(i,j) == 0) ? 0 : 1;
        }
      }
    }

    SVMParams params(1, 5e-2, 0.8, degrees);
    return params;
  }



}  // namespace obamadb