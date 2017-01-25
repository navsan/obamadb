#ifndef OBAMADB_MCTASK_H
#define OBAMADB_MCTASK_H

namespace obamadb {
/*
  // Matrix Completion parameters
  struct MCParams {
    MCParams(float mu,
             float step_size,
             float step_decay)
      : mu(mu),
        step_size(step_size),
        step_decay(step_decay),
        degrees_l(),
        degrees_r(){}

    float mu;
    float step_size;
    float step_decay;
    std::vector<int> degrees_l;
    std::vector<int> degrees_r;
  };

  class MCTask : MLTask {
  public:
    MCTask(DataView *dataView,
            fvector *sharedTheta,
            SVMParams *sharedParams)
      : MLTask(dataView),
        shared_theta_(sharedTheta),
        shared_params_(sharedParams) { }


    void execute(int thread_id, void* ml_state) override;

    Matrix * mat_l;
    Matrix * mat_r;
    MCParams* shared_params_;

    DISABLE_COPY_AND_ASSIGN(SVMTask);
  };
*/
} // namespace obamadb

#endif //OBAMADB_MCTASK_H
