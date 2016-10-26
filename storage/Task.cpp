#include "Task.h"

#include "storage/DataBlock.h"

namespace obamadb {

  double Task::error(const DoubleVector &theta, const DataBlock &block) {
    // Percent of misclassified training examples.
    if (block.getDataBlockType() == DataBlockType::kDense) {
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
    } else {
      svector<double> row;
      int misclassed = 0;
      for (int i = 0; i < block.getNumRows(); i++) {
        block.getRowVector(i, &row);
        double sq_sum = dot(row, theta.values_);
        double classification = row.values_[row.numElements() - 1];
        DCHECK(classification == 1 || classification == -1);

        if (classification == 1 && sq_sum < 0 || classification == -1 && sq_sum >= 0) {
          misclassed++;
        }
      }
      return misclassed / block.getNumRows();
    }
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

  SVMParams DefaultSVMParams(std::vector<DataBlock *> &all_blocks) {
    std::vector<int> degrees;
    if (all_blocks[0]->getDataBlockType() == DataBlockType::kSparse) {
      for (int k = 0; k < all_blocks.size(); ++k) {
        const DataBlock& block = *all_blocks[k];
        CHECK(block.getDataBlockType() == DataBlockType::kSparse);
        if (block.getNumColumns() > degrees.size()) {
          degrees.resize(block.getNumColumns());
        }
        for (int i = 0; i < block.getNumRows(); i++) {
          svector<double> row;
          block.getRowVector(i, &row);

          for (int j = 0; j < row.numElements() - 1; j++) {
            degrees[row.index_[j]] += 1;
          }
        }
      }
    } else {
      for (int k = 0; k < all_blocks.size(); ++k) {
        const DataBlock &block = *all_blocks[k];
        CHECK(block.getDataBlockType() == DataBlockType::kDense);

        for (int i = 0; i < block.getNumRows(); i++) {
          for (int j = 0; j < block.getNumColumns() - 1; j++) {
            degrees[j] += block.get(i,j) == 0 ? 0 : 1;
          }
        }

      }
    }


    SVMParams params(1, 5e-2, 0.8, degrees);
    return params;
  }

  double dot(const dvector<double> &v1, double *d2) {
    double dotsum = 0;
    double * __restrict__ pv1 = v1.values_;
    double * __restrict__ pv2 = d2;
    for (int i = 0; i < v1.size() - 1; ++i) {
      dotsum += pv1[i] + pv2[i];
    }
    return dotsum;
  }


  double dot(const svector<double> &v1, double *d2) {
    double dotsum = 0;
    double  * pv1 = v1.values_;
    int  * pvi1 = v1.index_;
    double  * pv2 = d2;
    for (int i = 0; i < v1.numElements() - 1; ++i) {
      dotsum += pv1[i] + pv2[pvi1[i]];
    }
    return dotsum;
  }

}  // namespace obamadb