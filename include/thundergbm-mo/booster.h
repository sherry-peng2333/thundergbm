//
// Created by ss on 19-1-17.
//

#ifndef THUNDERGBM_BOOSTER_H
#define THUNDERGBM_BOOSTER_H

#include <objective/objective_function.h>
#include <metric/metric.h>
#include <builder/function_builder.h>
#include <util/multi_device.h>
#include "common.h"
#include "syncarray.h"
#include "tree.h"
#include "row_sampler.h"

std::mutex mtx;

class Booster {
public:
    void init(const DataSet &dataSet, const GBMParam &param);

    void boost(vector<vector<Tree>> &boosted_model);

private:
    MSyncArray<GHPair> gradients;
    std::unique_ptr<ObjectiveFunction> obj;
    std::unique_ptr<Metric> metric;
    MSyncArray<float_type> y;
    std::unique_ptr<FunctionBuilder> fbuilder;
    RowSampler rowSampler;
    GBMParam param;
    int n_devices;
};

void Booster::init(const DataSet &dataSet, const GBMParam &param) {
    int n_available_device;
    cudaGetDeviceCount(&n_available_device);
    CHECK_GE(n_available_device, param.n_device) << "only " << n_available_device
                                            << " GPUs available; please set correct number of GPUs to use";
    this->param = param;
    fbuilder.reset(FunctionBuilder::create(param.tree_method));
    fbuilder->init(dataSet, param);
    obj.reset(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataSet);
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(param, dataSet);

    n_devices = param.n_device;
    int n_outputs = param.num_class * dataSet.n_instances_ * param.d_outputs_;
    gradients = MSyncArray<GHPair>(n_devices, n_outputs);
    if(param.objective.find("mo-reg:") != std::string::npos || param.objective.find("mo-lab:") != std::string::npos){
        y = MSyncArray<float_type>(n_devices, dataSet.n_instances_ * param.d_outputs_);
    }
    else{
        y = MSyncArray<float_type>(n_devices, dataSet.n_instances_ );
    }
    DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
        y[device_id].copy_from(dataSet.y.data(), dataSet.y.size());
    });
}

void Booster::boost(vector<vector<Tree>> &boosted_model) {
    TIMED_FUNC(timerObj);
    std::unique_lock<std::mutex> lock(mtx);

    //update gradients
    DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
        obj->get_gradient(y[device_id], fbuilder->get_y_predict()[device_id], gradients[device_id]);
    });
    if (param.bagging) rowSampler.do_bagging(gradients);
    PERFORMANCE_CHECKPOINT(timerObj);
    //build new model/approximate function
    boosted_model.push_back(fbuilder->build_approximate(gradients));

    PERFORMANCE_CHECKPOINT(timerObj);
    //show metric on training set
    LOG(INFO) << metric->get_name() << " = " << metric->get_score(fbuilder->get_y_predict().front());
}

#endif //THUNDERGBM_BOOSTER_H
