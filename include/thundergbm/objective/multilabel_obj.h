// Created by pxm on 22-11-09.

#ifndef THUNDERGBM_MULTILABEL_OBJ_H
#define THUNDERGBM_MULTILABEL_OBJ_H

#include "objective_function.h"
#include "thundergbm/util/device_lambda.cuh"

template<template<typename> class Loss>
class MultilabelObj : public ObjectiveFunction {
public:
    void get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p,
                      SyncArray<GHPair> &gh_pair) override {
        CHECK_EQ(y.size(), y_p.size())<<y.size() << "!=" << y_p.size();
        CHECK_EQ(y.size(), gh_pair.size());
        auto y_data = y.device_data();
        auto y_p_data = y_p.device_data();
        auto gh_pair_data = gh_pair.device_data();
        device_loop(y.size(), [=]__device__(int i) {
            gh_pair_data[i] = Loss<float_type>::gradient(y_data[i], y_p_data[i]);
        });
    }

    void predict_transform(SyncArray<float_type> &y) override {
        auto y_data = y.device_data();
        device_loop(y.size(), [=]__device__(int i) {
            y_data[i] = Loss<float_type>::predict_transform(y_data[i]);
        });
    }

    void configure(GBMParam param, const DataSet &dataset) override {}

    virtual ~MultilabelObj() override = default;

    string default_metric_name() override {
        return "mse";
    }
}; 

template<typename T>
struct SquareLoss {
    HOST_DEVICE static GHPair gradient(T y, T y_p) { return GHPair(y_p - y, 1); }

    HOST_DEVICE static T predict_transform(T x) { return x; }
};


#endif //THUNDERGBM_MULTILABEL_OBJ_H