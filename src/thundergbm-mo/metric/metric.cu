//
// Created by ss on 19-1-13.
//
#include "metric/metric.h"
#include "metric/pointwise_metric.h"
#include "metric/ranking_metric.h"
#include "metric/multiclass_metric.h"

Metric *Metric::create(string name) {
    if (name == "map") return new MAP;
    if (name == "rmse") return new RMSE;
    if (name == "ndcg") return new NDCG;
    if (name == "macc") return new MulticlassAccuracy;
    if (name == "momacc") return new MoMulticlassAccuracy;
    if (name == "error") return new BinaryClassMetric;
    LOG(FATAL) << "unknown metric " << name;
    return nullptr;
}

void Metric::configure(const GBMParam &param, const DataSet &dataset) {
    y.resize(dataset.y.size());
    y.copy_from(dataset.y.data(), dataset.y.size());
}
