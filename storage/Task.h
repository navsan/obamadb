#ifndef OBAMADB_TASK_H_
#define OBAMADB_TASK_H_

#include <iostream>
#include <mutex>

#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/LinearMath.h"

namespace obamadb {

  static std::mutex update_gradient;

  void rowGradient(
    double const *training_example,
    double y,
    double *theta,
    unsigned width,
    double num_training_examples);

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

    virtual void execute() {    };

    static double error(const DoubleVector& theta, const DataBlock& block);

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
      num_training_examples_(ntraining_examples) { }

    void execute() override {
      data_view_->reset();
      double *row = nullptr;
      while ((row = data_view_->getNext()) != nullptr) {
       // std::unique_lock<std::mutex> lk(update_gradient);
        rowGradient(row, row[model_->dimension_], model_->values_, model_->dimension_, num_training_examples_);
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

    SVMParams DefaultSVMParams(const std::vector<DataBlock*>& all_blocks);

  class SVMTask : public Task {
  public:
    SVMTask(DataView *dataView, DoubleVector *doubleVector, SVMParams& params)
      : Task(dataView, doubleVector), params_(params) {

    }

    void execute() override {
      data_view_->reset();
      double *example = nullptr;
      int itr = 0;
      while((example = data_view_->getNext()) != nullptr){
        itr++;
        double const y = example[model_->dimension_];
        double wxy = rowDot(example, model_->values_, model_->dimension_);
        wxy = wxy * y; // {-1, 1}
        // hinge active
        if (wxy < 1) {
          double const e = params_.step_size * y;
          // scale wieghts
          for (int i = 0; i < model_->dimension_; i++) {
            (*model_)[i] += example[i] * e;
          }
        }

        double const scalar = params_.step_size * params_.mu;
        for (int i = model_->dimension_; i-- > 0;) {
          double const deg = params_.degrees[i];
          model_->values_[i] *= 1 - scalar / deg;
        }
      }

      params_.step_size = params_.step_size * params_.step_decay;
    }

  private:

    SVMParams params_;
  };


}  // namespace obamadb

#endif //OBAMADB_TASK_H_
