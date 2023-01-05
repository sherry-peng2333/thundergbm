//
// Created by jiashuai on 19-1-23.
//

#include <thundergbm/builder/tree_builder.h>
#include "thundergbm/util/multi_device.h"
#include "thundergbm/util/device_lambda.cuh"

void TreeBuilder::update_tree() {
    TIMED_FUNC(timerObj);
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        auto& sp = this->sp[device_id];
        auto& tree = this->trees[device_id];
        auto sp_data = sp.host_data();
        LOG(DEBUG) << sp;
        int n_nodes_in_level = sp.size();

        Tree::TreeNode *nodes_data = tree.nodes.host_data();
        float_type rt_eps = param.rt_eps;
        float_type lambda = param.lambda;

        for(int i = 0; i < n_nodes_in_level; i++){
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
                lch.base_weight = SyncArray<float_type>(d_outputs_);
                rch.base_weight = SyncArray<float_type>(d_outputs_);
                lch.sum_gh_pair = SyncArray<GHPair>(d_outputs_);
                rch.sum_gh_pair = SyncArray<GHPair>(d_outputs_);
                node.split_feature_id = sp_data[i].split_fea_id;
                auto p_missing_gh = sp_data[i].fea_missing_gh.device_data();
                //todo process begin
                node.split_value = sp_data[i].fval;
                node.split_bid = sp_data[i].split_bid;
                rch.sum_gh_pair.copy_from(sp_data[i].rch_sum_gh);
                auto lsum_gh_pair_data = lch.sum_gh_pair.device_data();
                auto rsum_gh_pair_data = rch.sum_gh_pair.device_data();
                auto nsum_gh_pair_data = node.sum_gh_pair.device_data();
                if (sp_data[i].default_right) {
                    device_loop(d_outputs_, [=]__device__(int j){
                        rsum_gh_pair_data[j] = rsum_gh_pair_data[j] + p_missing_gh[j];
                    });  
                    node.default_right = true;
                }
                device_loop(d_outputs_, [=]__device__(int j){
                    lsum_gh_pair_data[j] = nsum_gh_pair_data[j] - rsum_gh_pair_data[j];
                });
                lch.calc_weight(lambda, d_outputs_);
                rch.calc_weight(lambda, d_outputs_);
            } else {
                //set leaf
                if (sp_data[i].nid == -1) return;
                int nid = sp_data[i].nid;
                Tree::TreeNode &node = nodes_data[nid];
                node.is_leaf = true;
                nodes_data[node.lch_index].is_valid = false;
                nodes_data[node.rch_index].is_valid = false;
            }
        }
        LOG(DEBUG) << tree.nodes;
    });
}

void TreeBuilder::predict_in_training(int k) {
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        auto y_predict_data = y_predict[device_id].device_data() + k * n_instances * d_outputs_;
        auto nid_data = ins2node_id[device_id].host_data();
        const Tree::TreeNode *nodes_data = trees[device_id].nodes.host_data();
        auto lr = param.learning_rate;
        int d_outputs_ = this->d_outputs_;
        for(int i = 0; i < n_instances; i++){
            int nid = nid_data[i];
            while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
            auto lr_nodes_data = nodes_data[nid].base_weight.device_data();
            device_loop(d_outputs_, [=]__device__(int j){
                y_predict_data[i * d_outputs_ + j] += lr * lr_nodes_data[j];
            });
        }
    });
}

void TreeBuilder::init(const DataSet &dataset, const GBMParam &param) {
    int n_available_device;
    cudaGetDeviceCount(&n_available_device);
    CHECK_GE(n_available_device, param.n_device) << "only " << n_available_device
                                                 << " GPUs available; please set correct number of GPUs to use";
    FunctionBuilder::init(dataset, param);      // this->param = param
    this->n_instances = dataset.n_instances();
    this->d_outputs_ = dataset.d_outputs_;
    trees = vector<Tree>(param.n_device);
    ins2node_id = MSyncArray<int>(param.n_device, n_instances);
    sp = MSyncArray<SplitPoint>(param.n_device);
    has_split = vector<bool>(param.n_device);
    int n_outputs = param.num_class * n_instances;
    y_predict = MSyncArray<float_type>(param.n_device, n_outputs);
    gradients = MSyncArray<GHPair>(param.n_device, n_instances*this->d_outputs_);
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
            if (!active_sp[global_pos]){
                global_sp_data[global_pos].gain = local_sp_data[j].gain;
                global_sp_data[global_pos].default_right = local_sp_data[j].default_right;
                global_sp_data[global_pos].nid = local_sp_data[j].nid;
                global_sp_data[global_pos].split_fea_id = local_sp_data[j].split_fea_id;
                global_sp_data[global_pos].fval = local_sp_data[j].fval;
                global_sp_data[global_pos].split_bid = local_sp_data[j].split_bid;
                global_sp_data[global_pos].fea_missing_gh.copy_from(local_sp_data[j].fea_missing_gh);
                global_sp_data[global_pos].rch_sum_gh.copy_from(local_sp_data[j].rch_sum_gh);}
            else if(global_sp_data[global_pos].gain < local_sp_data[j].gain){
                global_sp_data[global_pos].gain = local_sp_data[j].gain;
                global_sp_data[global_pos].default_right = local_sp_data[j].default_right;
                global_sp_data[global_pos].nid = local_sp_data[j].nid;
                global_sp_data[global_pos].split_fea_id = local_sp_data[j].split_fea_id;
                global_sp_data[global_pos].fval = local_sp_data[j].fval;
                global_sp_data[global_pos].split_bid = local_sp_data[j].split_bid;
                global_sp_data[global_pos].fea_missing_gh.copy_from(local_sp_data[j].fea_missing_gh);
                global_sp_data[global_pos].rch_sum_gh.copy_from(local_sp_data[j].rch_sum_gh);}
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
    for (int k = 0; k < param.tree_per_rounds; ++k) {
        Tree &tree = trees[k];
        DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
            this->ins2node_id[device_id].resize(n_instances);
            this->gradients[device_id].set_device_data(const_cast<GHPair *>(gradients[device_id].device_data() + k * n_instances * d_outputs_));
            this->trees[device_id].init2(this->gradients[device_id], param, this->d_outputs_);
        });
        for (int level = 0; level < param.depth; ++level) {
            DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
                find_split(level, device_id);
            });
            split_point_all_reduce(level);
            //LOG(INFO) << "split_point_all_reduce";
            {
                TIMED_SCOPE(timerObj, "apply sp");
                update_tree();
                //LOG(INFO) << "update_tree";
                update_ins2node_id();
                //LOG(INFO) << "update_ins2node_id";
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
        DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
            this->trees[device_id].prune_self(param.gamma);
        });
        predict_in_training(k);
        tree.nodes.resize(this->trees.front().nodes.size());
        tree.nodes.copy_from(this->trees.front().nodes);
        string s = tree.dump(param.depth);
        LOG(INFO) << "TREE:" << s;

    }
    LOG(INFO) << "one tree............";
    return trees;
}
