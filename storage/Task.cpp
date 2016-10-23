#include "Task.h"

namespace obamadb {

  double Task::error(const DoubleVector &theta, const DenseDataBlock &block) {
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

  SVMParams DefaultSVMParams(std::vector<DenseDataBlock *> &all_blocks) {
    int const dim = all_blocks[0]->getNumColumns() - 1;  // the assumption here is that the last element is that classification

    std::vector<int> degrees(dim);
    for (int k = 0; k < all_blocks.size(); ++k) {
      const DataBlock& block = *all_blocks[k];
      DCHECK_EQ(dim, block.getNumColumns() - 1);

      for (int i = 0; i < block.getNumRows(); i++) {
        for (int j = 0; j < dim; j++) {
          degrees[j] += block.get(i, j) != 0;
        }
      }
    }

    SVMParams params(1, 5e-2, 0.8, degrees);
    return params;
  }



}  // namespace obamadb