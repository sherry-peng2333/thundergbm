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
        float_type base_weight;
        int split_feature_id;
        float_type split_value;
        unsigned char split_bid;
        bool default_right;
        bool is_leaf;
        bool is_valid;// non-valid nodes are those that are "children" of leaf nodes
        bool is_pruned;// pruned after pruning

        GHPair sum_gh_pair;

        friend std::ostream &operator<<(std::ostream &os,
                                        const TreeNode &node);

        HOST_DEVICE void calc_weight(float_type lambda) {
            this->base_weight = -sum_gh_pair.g / (sum_gh_pair.h + lambda);
        }

        HOST_DEVICE bool splittable() const {
            return !is_leaf && is_valid;
        }

    };

    Tree() = default;

    Tree(const Tree &tree) {
        nodes.resize(tree.nodes.size());
        nodes.copy_from(tree.nodes);
        base_weight_mo.resize(tree.base_weight_mo.size());
        base_weight_mo.copy_from(tree.base_weight_mo);
        sum_gh_pair_mo.resize(tree.sum_gh_pair_mo.size());
        sum_gh_pair_mo.copy_from(tree.sum_gh_pair_mo);

    }

    Tree &operator=(const Tree &tree) {
        nodes.resize(tree.nodes.size());
        nodes.copy_from(tree.nodes);
        base_weight_mo.resize(tree.base_weight_mo.size());
        base_weight_mo.copy_from(tree.base_weight_mo);
        sum_gh_pair_mo.resize(tree.sum_gh_pair_mo.size());
        sum_gh_pair_mo.copy_from(tree.sum_gh_pair_mo);
        return *this;
    }

    void calc_weight_mo(float_type lambda, int nid);

    void init2(const SyncArray<GHPair> &gradients, const GBMParam &param);

    string dump(int depth) const;

    SyncArray<Tree::TreeNode> nodes;

    void prune_self(float_type gamma);

    int d_outputs_;
    SyncArray<float_type> base_weight_mo;
    SyncArray<GHPair> sum_gh_pair_mo;

private:
    void preorder_traversal(int nid, int max_depth, int depth, string &s) const;

    int try_prune_leaf(int nid, int np, float_type gamma, vector<int> &leaf_child_count);

    void reorder_nid();
};

#endif //THUNDERGBM_TREE_H
