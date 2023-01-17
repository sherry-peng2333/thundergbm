//
// Created by jiashuai on 18-1-18.
//

#ifndef THUNDERGBM_TREE_H
#define THUNDERGBM_TREE_H

#include "syncarray.h"
#include "sstream"


class Tree {
public:
    struct TreeNode {
        int final_id;// node id after pruning, may not equal to node index
        int lch_index;// index of left child
        int rch_index;// index of right child
        int parent_index;// index of parent node
        float_type gain;// gain of splitting this node
        SyncArray<float_type> base_weight;
        int split_feature_id;
        float_type split_value;
        unsigned char split_bid;
        bool default_right;
        bool is_leaf;
        bool is_valid;// non-valid nodes are those that are "children" of leaf nodes
        bool is_pruned;// pruned after pruning

        //GHPair sum_gh_pair;
        SyncArray<GHPair> sum_gh_pair;

        friend std::ostream &operator<<(std::ostream &os,
                                        const TreeNode &node);

        HOST_DEVICE void calc_weight(float_type lambda, int dimension) {
            auto sum_gh_pair_data = sum_gh_pair.host_data();
            auto base_weight_data = base_weight.host_data();
            for(int i = 0; i < dimension; i++){
                base_weight_data[i] = -sum_gh_pair_data[i].g / (sum_gh_pair_data[i].h + lambda);
            }
        }

        HOST_DEVICE bool splittable() const {
            return !is_leaf && is_valid;
        }

//        Tree::TreeNode &operator=(const Tree::TreeNode &treeNode){
//            this->final_id = treeNode.final_id;
//            this->lch_index = treeNode.lch_index;
//            this->rch_index = treeNode.rch_index;
//            this->parent_index = treeNode.parent_index;
//            this->gain = treeNode.gain;
//            this->split_feature_id = treeNode.split_feature_id;
//            this->split_value = treeNode.split_value;
//            this->split_bid = treeNode.split_bid;
//            this->default_right = treeNode.default_right;
//            this->is_leaf = treeNode.is_leaf;
//            this->is_valid = treeNode.is_valid;
//            this->is_pruned = treeNode.is_pruned;
//            int d_outputs_ = treeNode.base_weight.size();
//            this->base_weight = SyncArray<float_type>(d_outputs_);
//            this->base_weight.copy_from(treeNode.base_weight);
//            // to do: add sum_gh_pair copy is necessary
//            return *this;
//        }

    };

    Tree() = default;

    Tree(const Tree &tree) {
        this->d_outputs_ = tree.d_outputs_;
        nodes.resize(tree.nodes.size());
        nodes.copy_from(tree.nodes);
//        auto nodes_data = nodes.host_data();
//        auto tree_nodes_data = tree.nodes.host_data();
//        for(int i = 0; i < tree.nodes.size(); i++){
//            nodes_data[i].base_weight = SyncArray<float_type>(d_outputs_);
//            nodes_data[i].base_weight.copy_from(tree_nodes_data->base_weight);
//        }
    }

    Tree &operator=(const Tree &tree) {
        this->d_outputs_ = tree.d_outputs_;
        nodes.resize(tree.nodes.size());
        nodes.copy_from(tree.nodes);
//        auto nodes_data = nodes.host_data();
//        auto tree_nodes_data = tree.nodes.host_data();
//        for(int i = 0; i < tree.nodes.size(); i++){
//            nodes_data[i].base_weight = SyncArray<float_type>(d_outputs_);
//            nodes_data[i].base_weight.copy_from(tree_nodes_data->base_weight);
//        }
        return *this;
    }

    void init2(SyncArray<GHPair> &gradients, const GBMParam &param, int d_outputs_=1);

    string dump(int depth) const;

    SyncArray<Tree::TreeNode> nodes;

    int d_outputs_;

    void prune_self(float_type gamma);

private:
    void preorder_traversal(int nid, int max_depth, int depth, string &s) const;

    int try_prune_leaf(int nid, int np, float_type gamma, vector<int> &leaf_child_count);

    void reorder_nid();
};

#endif //THUNDERGBM_TREE_H
