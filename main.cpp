#include "storage/Loader.h"
#include "storage/DataBlock.h"

#include <iostream>
#include <string>
#include <memory>
#include <glog/logging.h>

namespace obamadb {

  namespace {
    static const double alpha = 0.00001;

    double error(DataBlock * const A,
                 DataBlock * const y,
                 double const * theta) {
      DCHECK_EQ(A->getNumRows(), y->getNumRows());

      const unsigned N = y->getNumRows();
      const unsigned n_theta = A->getWidth();
      double *a_cursor = A->getStore();
      double *y_cursor = y->getStore();
      long double e_sum = 0;
      for (unsigned i = 0; i < N; ++i) {
        long double a_theta = 0;
        for (unsigned j = 0; j < n_theta; ++j) {
          a_theta += theta[j] * *a_cursor;
          a_cursor += 1;
        }
        // Round
        if (a_theta < 0) {
          a_theta = 0.0;
        } else {
          a_theta = 1.0;
        }
        long double r = *y_cursor - a_theta;
        e_sum += r * r;
        y_cursor++;
      }
      return e_sum / N;
    }

    double rowDot(
      double const * row_a,
      double const * row_b,
      unsigned row_dimension){

      double sum = 0.0;
      for (unsigned col = 0; col < row_dimension; ++col) {
        sum += row_a[col] * row_b[col];
      }
      return sum;
    }

    /**
     * Applies the gradient function of one training example on theta.
     *
     * @param training_example
     * @param y
     * @param theta
     * @param width
     * @param num_training_examples
     * @param delta
     */
    void rowGradient(
      double const * training_example,
      double y,
      double * theta,
      unsigned width,
      double num_training_examples) {

      double residual = y - rowDot(training_example, theta, width);
      double train_factor = (alpha * 2.0) / num_training_examples;
      for (unsigned col = 0; col < width; ++col) {
        theta[col] += train_factor * residual * training_example[col];
      }
    }

    /**
     * Runs gradient descent over the entire dataset for one iteration.
     * @param A
     * @param y
     * @param theta
     * @return
     */
    double* gradientItr(
                      DataBlock const * A,
                      DataBlock const * y,
                      double * theta) {
      for (unsigned row = 0; row < A->getNumRows(); ++row) {
        rowGradient(A->getRow(row), y->get(row, 0), theta, A->getWidth(), A->getNumRows());
      }
      return theta;
    }

  }


  int main() {
    std::unique_ptr<DataBlock> iris_data(Loader::load("storage/iris.dat"));

    // Get data with either a 1 or 0 value.
    std::function<bool(double)> select_fn = [] (double input) {
      return input == 0.0 || input == 1.0;
    };
    std::unique_ptr<DataBlock> two_flower_data(iris_data->filter(select_fn, 0));
    std::unique_ptr<DataBlock> y_vals(two_flower_data->slice(0, 1));

    // Pad A values with 1s.
    std::unique_ptr<DataBlock> A_vals(two_flower_data->slice(0, two_flower_data->getWidth()));
    double* pad_ptr = A_vals->getStore();
    for (unsigned i = 0; i < A_vals->getNumRows(); ++i) {
      *pad_ptr = 1;
      pad_ptr += A_vals->getWidth();
    }

    std::cout << "y vals:" << std::endl;
    std::cout << *y_vals << std::endl;

    std::cout << "A vals:" << std::endl;
    std::cout << *A_vals << std::endl;

    // Initialize theta
    double *theta = new double[A_vals->getWidth()];
    theta[0] = 0.01;
    theta[1] = -0.01;
    theta[2] = 0.2;
//    for (unsigned i = 0; i < A_vals->getWidth(); i++) {
//      theta[i] = 0.2;
//    }

    std::cout << "E: " << error(A_vals.get(), y_vals.get(), theta) << std::endl;

    for (unsigned i = 0; i < 10000; ++i) {
      gradientItr(A_vals.get(),y_vals.get(),theta);
      double er = error(A_vals.get(), y_vals.get(), theta);
      std::cout << "E: " << er << std::endl;
      if (er == 0)
        break;
    }

    for (unsigned i = 0; i < A_vals->getWidth(); i++) {
      std::cout << std::to_string(theta[i]) << std::endl;
    }

    delete theta;
    return 0;
  }
} // namespace obamadb

int main() {
  obamadb::main();
}