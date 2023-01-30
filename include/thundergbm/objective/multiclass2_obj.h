////
//// Created by xuemei on 23-1-18.
////
//
//#ifndef THUNDERGBM_MULTICLASS2_OBJ_H
//#define THUNDERGBM_MULTICLASS2_OBJ_H
//
//#include "objective_function.h"
//class Ce : public ObjectiveFunction {
//public:
//    void get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p,
//                      SyncArray<GHPair> &gh_pair) override;
//
//    void predict_transform(SyncArray<float_type> &y) override;
//
//    void configure(GBMParam param, const DataSet &dataset) override;
//
//    string default_metric_name() override { return "macc"; }
//
//    virtual ~Softmax() override = default;
//
//protected:
//    int num_class;
//    SyncArray<float_type> label;
//};
//
//#endif //THUNDERGBM_MULTICLASS2_OBJ_H
