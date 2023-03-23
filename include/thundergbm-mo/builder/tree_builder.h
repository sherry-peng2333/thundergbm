//
// Created by jiashuai on 19-1-23.
//

#ifndef THUNDERGBM_TREEBUILDER_H
#define THUNDERGBM_TREEBUILDER_H

#include <tree.h>
#include "common.h"
#include "shard.h"
#include "function_builder.h"


class TreeBuilder : public FunctionBuilder {
public:
    virtual void find_split(int level, int device_id) = 0;
    virtual void find_split_mo(int level, int device_id) = 0;

    virtual void update_ins2node_id() = 0;

    vector<Tree> build_approximate(const MSyncArray<GHPair> &gradients) override;

    void init(const DataSet &dataset, const GBMParam &param) override;

    virtual void update_tree();

    void predict_in_training(int k);

    virtual void split_point_all_reduce(int depth);

    virtual void ins2node_id_all_reduce(int depth);

    virtual ~TreeBuilder(){};

protected:
    vector<Shard> shards;
    int n_instances;
    int d_outputs_;                         // for multi-outputs
    bool multi_outputs;                     // for multi-outputs
    vector<Tree> trees;
    MSyncArray<int> ins2node_id;
    MSyncArray<SplitPoint> sp;
    MSyncArray<GHPair> sp_fea_missing_gh;   // for multi-outputs
    MSyncArray<GHPair> sp_rch_sum_gh;       // for multi-outputs
    MSyncArray<GHPair> gradients;
    vector<bool> has_split;
    float_type build_hist_time;
    float_type subtract_time;
};


#endif //THUNDERGBM_TREEBUILDER_H
