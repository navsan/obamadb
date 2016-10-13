#include "storage/DataBlock.h"
#include "storage/LinearMath.h"
#include "storage/Loader.h"

#include <iostream>
#include <string>
#include <memory>

#include <glog/logging.h>

namespace obamadb {

  int main() {
    auto blocks = Loader::load("storage/iris.dat");
    CHECK_LT(0, blocks.size());
    std::unique_ptr<DataBlock> iris_data(blocks[0]);

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
      gradientItr(A_vals.get(), y_vals.get(), theta);
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