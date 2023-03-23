//
// Created by jiashuai on 19-1-23.
//

#include <builder/tree_builder.h>
#include "util/multi_device.h"
#include "util/device_lambda.cuh"

void TreeBuilder::update_tree() {
    TIMED_FUNC(timerObj);
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        auto& sp = this->sp[device_id];
        auto& tree = this->trees[device_id];
        SyncArray<GHPair> &sp_fea_missing_gh = this->sp_fea_missing_gh[device_id];
        SyncArray<GHPair> &sp_rch_sum_gh = this->sp_rch_sum_gh[device_id];
        auto sp_data = sp.device_data();
        LOG(DEBUG) << sp;
        int n_nodes_in_level = sp.size();

        Tree::TreeNode *nodes_data = tree.nodes.device_data();
        float_type rt_eps = param.rt_eps;
        float_type lambda = param.lambda;
        if(multi_outputs){
            int d_outputs = this->d_outputs_;
            GHPair *sum_gh_pair_mo_data = tree.sum_gh_pair_mo.device_data();
            float_type *base_weight_mo_data = tree.base_weight_mo.device_data();
            auto sp_fea_missing_gh_data = sp_fea_missing_gh.device_data();
            auto sp_rch_sum_gh_data = sp_rch_sum_gh.device_data();
            // when it is multi_outputs
            device_loop(n_nodes_in_level, [=]__device__(int i) {
                float_type best_split_gain = sp_data[i].gain;
                if (best_split_gain > rt_eps) {
                    //do split
                    if (sp_data[i].nid == -1) return;
                    int nid = sp_data[i].nid;
                    Tree::TreeNode &node = nodes_data[nid];
                    node.gain = best_split_gain;
                    int lid = node.lch_index;
                    int rid = node.rch_index;
                    Tree::TreeNode &lch = nodes_data[lid];//left child
                    Tree::TreeNode &rch = nodes_data[rid];//right child
                    lch.is_valid = true;
                    rch.is_valid = true;
                    node.split_feature_id = sp_data[i].split_fea_id;
                    //todo process begin
                    node.split_value = sp_data[i].fval;
                    node.split_bid = sp_data[i].split_bid;
                    for(int j = 0; j < d_outputs; j++){
                        sum_gh_pair_mo_data[rid*d_outputs+j] = sp_rch_sum_gh_data[i*d_outputs+j];
                    }
                    if (sp_data[i].default_right) {
                        for(int j = 0; j < d_outputs; j++){
                            sum_gh_pair_mo_data[rid*d_outputs+j] = sum_gh_pair_mo_data[rid*d_outputs+j] + sp_fea_missing_gh_data[i*d_outputs+j];
                        }
                        node.default_right = true;
                    }
                    for(int j = 0; j < d_outputs; j++){
                        sum_gh_pair_mo_data[lid*d_outputs+j] = sum_gh_pair_mo_data[nid*d_outputs+j] - sum_gh_pair_mo_data[rid*d_outputs+j];
                    }
                    for(int j = 0; j < d_outputs; j++){
                        base_weight_mo_data[lid*d_outputs+j] = -sum_gh_pair_mo_data[lid*d_outputs+j].g / (sum_gh_pair_mo_data[lid*d_outputs+j].h + lambda);
                        base_weight_mo_data[rid*d_outputs+j] = -sum_gh_pair_mo_data[rid*d_outputs+j].g / (sum_gh_pair_mo_data[rid*d_outputs+j].h + lambda);
                    }
                } else {
                    //set leaf
                    if (sp_data[i].nid == -1) return;
                    int nid = sp_data[i].nid;
                    Tree::TreeNode &node = nodes_data[nid];
                    node.is_leaf = true;
                    nodes_data[node.lch_index].is_valid = false;
                    nodes_data[node.rch_index].is_valid = false;
                }
            });
        }
        else{
            device_loop(n_nodes_in_level, [=]__device__(int i) {
                float_type best_split_gain = sp_data[i].gain;
                if (best_split_gain > rt_eps) {
                    //do split
                    if (sp_data[i].nid == -1) return;
                    int nid = sp_data[i].nid;
                    Tree::TreeNode &node = nodes_data[nid];
                    node.gain = best_split_gain;

                    Tree::TreeNode &lch = nodes_data[node.lch_index];//left child
                    Tree::TreeNode &rch = nodes_data[node.rch_index];//right child
                    lch.is_valid = true;
                    rch.is_valid = true;
                    node.split_feature_id = sp_data[i].split_fea_id;
                    GHPair p_missing_gh = sp_data[i].fea_missing_gh;
                    //todo process begin
                    node.split_value = sp_data[i].fval;
                    node.split_bid = sp_data[i].split_bid;
                    rch.sum_gh_pair = sp_data[i].rch_sum_gh;
                    if (sp_data[i].default_right) {
                        rch.sum_gh_pair = rch.sum_gh_pair + p_missing_gh;
                        node.default_right = true;
                    }
                    lch.sum_gh_pair = node.sum_gh_pair - rch.sum_gh_pair;
                    lch.calc_weight(lambda);
                    rch.calc_weight(lambda);
                } else {
                    //set leaf
                    if (sp_data[i].nid == -1) return;
                    int nid = sp_data[i].nid;
                    Tree::TreeNode &node = nodes_data[nid];
                    node.is_leaf = true;
                    nodes_data[node.lch_index].is_valid = false;
                    nodes_data[node.rch_index].is_valid = false;
                }
            });
        }
        LOG(DEBUG) << tree.nodes;
    });
}

void TreeBuilder::predict_in_training(int k) {
    if(multi_outputs){
        DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
            auto y_predict_data = y_predict[device_id].device_data() + k * n_instances;
            auto nid_data = ins2node_id[device_id].device_data();
            const Tree::TreeNode *nodes_data = trees[device_id].nodes.device_data();
            auto base_weight_data = trees[device_id].base_weight_mo.device_data();
            int d_outputs_ = this->d_outputs_;
            auto lr = param.learning_rate;
            device_loop(n_instances, [=]__device__(int i) {
                int nid = nid_data[i];
                while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
                for(int j = 0; j < d_outputs_; j++){
                    y_predict_data[i*d_outputs_+j] += lr * base_weight_data[nid*d_outputs_+j];
                }
            });
        });
    }
    else{
        DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
            auto y_predict_data = y_predict[device_id].device_data() + k * n_instances;
            auto nid_data = ins2node_id[device_id].device_data();
            const Tree::TreeNode *nodes_data = trees[device_id].nodes.device_data();
            auto lr = param.learning_rate;
            device_loop(n_instances, [=]__device__(int i) {
                int nid = nid_data[i];
                while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
                y_predict_data[i] += lr * nodes_data[nid].base_weight;
            });
        });
    }

}

void TreeBuilder::init(const DataSet &dataset, const GBMParam &param) {
    int n_available_device;
    cudaGetDeviceCount(&n_available_device);
    CHECK_GE(n_available_device, param.n_device) << "only " << n_available_device
                                                 << " GPUs available; please set correct number of GPUs to use";
    FunctionBuilder::init(dataset, param);
    this->n_instances = dataset.n_instances_;
    this->d_outputs_ = dataset.d_outputs_;
    this->multi_outputs = param.multi_outputs;
    trees = vector<Tree>(param.n_device);
    ins2node_id = MSyncArray<int>(param.n_device, n_instances);
    sp = MSyncArray<SplitPoint>(param.n_device);
    has_split = vector<bool>(param.n_device);
    int n_outputs = param.num_class * n_instances * param.d_outputs_;
    y_predict = MSyncArray<float_type>(param.n_device, n_outputs);
    gradients = MSyncArray<GHPair>(param.n_device, n_instances * param.d_outputs_);
    if(param.multi_outputs){
        sp_fea_missing_gh = MSyncArray<GHPair>(param.n_device);
        sp_rch_sum_gh = MSyncArray<GHPair>(param.n_device);
    }
}

void TreeBuilder::ins2node_id_all_reduce(int depth) {
    //get global ins2node id
    {
        SyncArray<int> local_ins2node_id(n_instances);
        auto local_ins2node_id_data = local_ins2node_id.device_data();
        auto global_ins2node_id_data = ins2node_id.front().device_data();
        for (int d = 1; d < param.n_device; d++) {
            local_ins2node_id.copy_from(ins2node_id[d]);
            device_loop(n_instances, [=]__device__(int i) {
                global_ins2node_id_data[i] = (global_ins2node_id_data[i] > local_ins2node_id_data[i]) ?
                                             global_ins2node_id_data[i] : local_ins2node_id_data[i];
            });
        }
    }
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        ins2node_id[device_id].copy_from(ins2node_id.front());
    });
}

void TreeBuilder::split_point_all_reduce(int depth) {
    TIMED_FUNC(timerObj);
    //get global best split of each node
    int n_nodes_in_level = 1 << depth;//2^i
    int nid_offset = (1 << depth) - 1;//2^i - 1
    auto global_sp_data = sp.front().host_data();
    vector<bool> active_sp(n_nodes_in_level);

    for (int device_id = 0; device_id < param.n_device; device_id++) {
        auto local_sp_data = sp[device_id].host_data();
        for (int j = 0; j < sp[device_id].size(); j++) {
            int sp_nid = local_sp_data[j].nid;
            if (sp_nid == -1) continue;
            int global_pos = sp_nid - nid_offset;
            if (!active_sp[global_pos])
                global_sp_data[global_pos] = local_sp_data[j];
            else
                global_sp_data[global_pos] = (global_sp_data[global_pos].gain >= local_sp_data[j].gain)
                                             ?
                                             global_sp_data[global_pos] : local_sp_data[j];
            active_sp[global_pos] = true;
        }
    }
    //set inactive sp
    for (int n = 0; n < n_nodes_in_level; n++) {
        if (!active_sp[n])
            global_sp_data[n].nid = -1;
    }
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
       sp[device_id].copy_from(sp.front());
    });
    LOG(DEBUG) << "global best split point = " << sp.front();
}

vector<Tree> TreeBuilder::build_approximate(const MSyncArray<GHPair> &gradients) {
    vector<Tree> trees(param.tree_per_rounds);
    TIMED_FUNC(timerObj);
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        this->shards[device_id].column_sampling(param.column_sampling_rate);
    });
    float_type build_hist_total_time = 0;
    float_type subtract_time = 0;
    for (int k = 0; k < param.tree_per_rounds; ++k) {
        Tree &tree = trees[k];
        DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
            this->ins2node_id[device_id].resize(n_instances);
            this->gradients[device_id].set_device_data(const_cast<GHPair *>(gradients[device_id].device_data() + k * n_instances));
            this->trees[device_id].init2(this->gradients[device_id], param);
        });

        for (int level = 0; level < param.depth; ++level) {
            DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
                find_split_mo(level, device_id);
            });
            split_point_all_reduce(level);
            {
                TIMED_SCOPE(timerObj, "apply sp");
                update_tree();
                update_ins2node_id();
                {
                    LOG(TRACE) << "gathering ins2node id";
                    //get final result of the reset instance id to node id
                    bool has_split = false;
                    for (int d = 0; d < param.n_device; d++) {
                        has_split |= this->has_split[d];
                    }
                    if (!has_split) {
                        LOG(INFO) << "no splittable nodes, stop";
                        break;
                    }
                }
                ins2node_id_all_reduce(level);
            }
        }
//        DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
//            this->trees[device_id].prune_self(param.gamma);
//        });
        predict_in_training(k);
        tree.nodes.resize(this->trees.front().nodes.size());
        tree.nodes.copy_from(this->trees.front().nodes);
        tree.sum_gh_pair_mo.resize(this->trees.front().sum_gh_pair_mo.size());
        tree.sum_gh_pair_mo.copy_from(this->trees.front().sum_gh_pair_mo);
        tree.base_weight_mo.resize(this->trees.front().base_weight_mo.size());
        tree.base_weight_mo.copy_from(this->trees.front().base_weight_mo);
        tree.d_outputs_ = this->trees.front().d_outputs_;
        build_hist_total_time += this->build_hist_time;
        subtract_time += this->subtract_time;
//        string s = tree.dump(param.depth);
//        LOG(INFO) << "TREE:" << s;
    }
    LOG(INFO) << "time for atom operation of building hist: " << build_hist_total_time;
    LOG(INFO) << "time for subtract operation: " << subtract_time;
    return trees;
}
