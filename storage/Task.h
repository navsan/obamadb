#ifndef OBAMADB_TASK_H_
#define OBAMADB_TASK_H_

#include <iostream>
#include <mutex>

#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/LinearMath.h"
#include "storage/Utils.h"

namespace obamadb {

  static std::mutex update_gradient;

  void rowGradient(
    double const *training_example,
    double y,
    double *theta,
    unsigned width,
    double num_training_examples);

  double dot(const dvector<double>& v1, double * d2);

  double dot(const svector<double>& v1, double * d2);

  /*
   * The Task describes what a thread will perform.
   */
  class Task {
  public:
    /*
     * Does not take ownership of the view or model.
     */
    Task(DataView *dataView, DoubleVector *doubleVector)
      : data_view_(dataView), model_(doubleVector) {

    }

    ~Task() {};

    virtual void execute() {};

    static double error(const DoubleVector &theta, const DataBlock &block);

  protected:
    DataView *data_view_;
    DoubleVector *model_;
  };

  class LinearRegressionTask : public Task {
  public:
    LinearRegressionTask(DataView *dataView,
                         DoubleVector *doubleVector,
                         double ntraining_examples)
      : Task(dataView, doubleVector),
        num_training_examples_(ntraining_examples) {}

    void execute() override {
      data_view_->reset();
      dvector<double> row;
      while ((data_view_->getNext(&row))) {
        DCHECK_EQ(model_->dimension_, row.size() - 1);

        double y = *row.getLast();
        double residual = y - dot(row, model_->values_);
        double train_factor = (alpha * 2.0) / model_->dimension_;
        for (unsigned col = 0; col < row.size() - 1; ++col) {
          model_->values_[col] += train_factor * residual * *row.get(col);
        }
      }
    }

    double num_training_examples_;
  };

  struct SVMParams {
    SVMParams(double mu, double step_size, double step_decay, std::vector<int> degrees)
      : mu(mu), step_size(step_size), step_decay(step_decay), degrees(degrees) {}

    double mu;
    double step_size;
    double step_decay;
    std::vector<int> degrees;
  };

  SVMParams DefaultSVMParams(std::vector<DataBlock *> &all_blocks);


  class SVMTask : public Task {
  public:
    SVMTask(DataView *dataView, DoubleVector *doubleVector, SVMParams &params)
      : Task(dataView, doubleVector), params_(params) {

    }

    void execute() override {
      data_view_->reset();
      svector<double> row;
      while (data_view_->getNext(&row)) {
        double const y = *row.getLast();
        double wxy = dot(row, model_->values_);
        wxy = wxy * y; // {-1, 1}
        // hinge active
        if (wxy < 1) {
          double const e = params_.step_size * y;
          // scale wieghts
          for (int i = 0; i < row.numElements() - 1; i++) {
            (*model_)[i] += row.values_[row.index_[i]] * e;
          }
        }

        double const scalar = params_.step_size * params_.mu;
        // scale only the values which were updated.
        for (int i = row.numElements() - 1; i-- > 0;) {
          const int idx_j = row.index_[i];
          double const deg = params_.degrees[idx_j];
          model_->values_[idx_j] *= 1 - scalar / deg;
        }
      }

      params_.step_size = params_.step_size * params_.step_decay;
    }

  private:

    SVMParams params_;
  };


}  // namespace obamadb

#endif //OBAMADB_TASK_H_
