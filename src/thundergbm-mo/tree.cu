//
// Created by jiashuai on 18-1-18.
//
#include "tree.h"
#include "util/device_lambda.cuh"
#include "thrust/reduce.h"

void Tree::init2(const SyncArray<GHPair> &gradients, const GBMParam &param) {
    TIMED_FUNC(timerObj);
    int n_max_nodes = static_cast<int>(pow(2, param.depth + 1) - 1);
    nodes = SyncArray<TreeNode>(n_max_nodes);
    auto node_data = nodes.device_data();
    device_loop(n_max_nodes, [=]__device__(int i) {
        node_data[i].final_id = i;
        node_data[i].split_feature_id = -1;
        node_data[i].is_valid = false;
        node_data[i].parent_index = i == 0 ? -1 : (i - 1) / 2;
        if (i < n_max_nodes / 2) {
            node_data[i].is_leaf = false;
            node_data[i].lch_index = i * 2 + 1;
            node_data[i].rch_index = i * 2 + 2;
        } else {
            //leaf nodes
            node_data[i].is_leaf = true;
            node_data[i].lch_index = -1;
            node_data[i].rch_index = -1;
        }
    });

    //init root node
    if(!param.multi_outputs){
        GHPair sum_gh = thrust::reduce(thrust::cuda::par, gradients.device_data(), gradients.device_end());
        float_type lambda = param.lambda;
        device_loop<1, 1>(1, [=]__device__(int i) {
            Tree::TreeNode &root_node = node_data[0];
            root_node.sum_gh_pair = sum_gh;
            root_node.is_valid = true;
            root_node.calc_weight(lambda);
        });
    }
    else{
        d_outputs_ = param.d_outputs_;
        base_weight_mo = SyncArray<float_type>(d_outputs_*n_max_nodes);
        sum_gh_pair_mo = SyncArray<GHPair>(d_outputs_*n_max_nodes);
        auto base_weight_data = base_weight_mo.device_data();
        auto sum_gh_pair_data = sum_gh_pair_mo.device_data();
        auto gradients_data = gradients.device_data();
        int gradients_size = gradients.size();
        CHECK_EQ(gradients_size%d_outputs_, 0);
        int d_outputs_ = param.d_outputs_;
        device_loop(gradients_size, [=]__device__(int i){
            int id = i % d_outputs_;
            // todo: improve atomicadd
            const GHPair src = gradients_data[i];
            GHPair &dest = sum_gh_pair_data[id];
            if(src.h != 0)
                atomicAdd(&dest.h, src.h);
            if(src.g != 0)
                atomicAdd(&dest.g, src.g);
        });
        float_type lambda = param.lambda;
        calc_weight_mo(lambda, 0);
        device_loop<1, 1>(1, [=]__device__(int i) {
            Tree::TreeNode &root_node = node_data[0];
            root_node.is_valid = true;
        });

    }
}

string Tree::dump(int depth) const {
    string s("\n");
    preorder_traversal(0, depth, 0, s);
    return s;
}

void Tree::preorder_traversal(int nid, int max_depth, int depth, string &s) const {
    if(nid == -1)//child of leaf node
        return;
    const TreeNode &node = nodes.host_data()[nid];
    const TreeNode *node_data = nodes.host_data();
    const float_type *base_weight_mo_data = base_weight_mo.host_data();
    if (node.is_valid && !node.is_pruned) {
        s = s + string(static_cast<unsigned long>(depth), '\t');

        if(node.is_leaf){
            s = s + string_format("%d:leaf=", node.final_id);
            for(int i = 0; i < d_outputs_; i++){
                s = s + string_format("%.6g  ", base_weight_mo_data[nid*d_outputs_+i]);
            }
            s = s + "\n";
        }
        else {
            int lch_final_id = node_data[node.lch_index].final_id;
            int rch_final_id = node_data[node.rch_index].final_id;
            string str_inter_node = string_format("%d:[f%d<%.6g] yes=%d,no=%d,missing=%d\n", node.final_id,
                                                  node.split_feature_id + 1,
                                                  node.split_value, lch_final_id, rch_final_id,
                                                  node.default_right == 0 ? lch_final_id : rch_final_id);
            s = s + str_inter_node;
        }
//             string_format("%d:[f%d<%.6g], weight=%f, gain=%f, dr=%d\n", node.final_id, node.split_feature_id + 1,
//                           node.split_value,
//                           node.base_weight, node.gain, node.default_right));
    }
    if (depth < max_depth) {
        preorder_traversal(node.lch_index, max_depth, depth + 1, s);
        preorder_traversal(node.rch_index, max_depth, depth + 1, s);
    }
}

std::ostream &operator<<(std::ostream &os, const Tree::TreeNode &node) {
    os << string_format("\nnid:%d,l:%d,v:%d,split_feature_id:%d,f:%f,gain:%f,r:%d,w:%f,", node.final_id, node.is_leaf,
                        node.is_valid,
                        node.split_feature_id, node.split_value, node.gain, node.default_right, node.base_weight);
    os << "g/h:" << node.sum_gh_pair;
    return os;
}

void Tree::reorder_nid() {
    int nid = 0;
    Tree::TreeNode *nodes_data = nodes.host_data();
    for (int i = 0; i < nodes.size(); ++i) {
        if (nodes_data[i].is_valid && !nodes_data[i].is_pruned) {
            nodes_data[i].final_id = nid;
            nid++;
        }
    }
}

int Tree::try_prune_leaf(int nid, int np, float_type gamma, vector<int> &leaf_child_count) {
    Tree::TreeNode *nodes_data = nodes.host_data();
    int p_nid = nodes_data[nid].parent_index;
    if (p_nid == -1) return np;// is root
    Tree::TreeNode &p_node = nodes_data[p_nid];
    Tree::TreeNode &lch = nodes_data[p_node.lch_index];
    Tree::TreeNode &rch = nodes_data[p_node.rch_index];
    leaf_child_count[p_nid]++;
    if (leaf_child_count[p_nid] >= 2 && p_node.gain < gamma) {
        //do pruning
        //delete two children
        CHECK(lch.is_leaf);
        CHECK(rch.is_leaf);
        lch.is_pruned = true;
        rch.is_pruned = true;
        //make parent to leaf
        p_node.is_leaf = true;
        return try_prune_leaf(p_nid, np + 2, gamma, leaf_child_count);
    } else return np;
}

void Tree::prune_self(float_type gamma) {
    vector<int> leaf_child_count(nodes.size(), 0);
    Tree::TreeNode *nodes_data = nodes.host_data();
    int n_pruned = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        if (nodes_data[i].is_leaf && nodes_data[i].is_valid) {
            n_pruned = try_prune_leaf(i, n_pruned, gamma, leaf_child_count);
        }
    }
    LOG(DEBUG) << string_format("%d nodes are pruned", n_pruned);
    reorder_nid();
}

void Tree::calc_weight_mo(float_type lambda, int nid) {
    auto sum_gh_pair_data = sum_gh_pair_mo.device_data();
    auto base_weight_data = base_weight_mo.device_data();
    int d_outputs_ = this->d_outputs_;
    device_loop(d_outputs_, [=]__device__(int did){
        base_weight_data[nid*d_outputs_+did] = -sum_gh_pair_data[nid*d_outputs_+did].g / (sum_gh_pair_data[nid*d_outputs_+did].h + lambda);
    });
}
