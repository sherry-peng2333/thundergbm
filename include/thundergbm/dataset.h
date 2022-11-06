//
// Created by jiashuai on 18-1-17.
//

#ifndef THUNDERGBM_DATASET_H
#define THUNDERGBM_DATASET_H

#include "common.h"
#include "syncarray.h"

class DataSet {
public:
    ///load dataset from file
    void load_from_sparse(int n_instances, float *csr_val, int *csr_row_ptr, int *csr_col_idx, float *y,
            int *group, int num_group, GBMParam &param);
    void load_from_file(string file_name, GBMParam &param);
    void load_from_file_mo(string file_name, GBMParam &param);
    void load_csc_from_file(string file_name, GBMParam &param, int const nfeatures=500);
    void load_group_file(string file_name);
    void group_label();

    size_t n_features() const;
    size_t n_instances() const;
    size_t n_instances_mo() const;

    vector<float_type> csr_val;           // csr: compressed sparse row format
    vector<int> csr_row_ptr;
    vector<int> csr_col_idx;
    vector<float_type> y;
    size_t n_features_;                   // number of features   
    vector<int> group;
    vector<float_type> label;

    // for multi-outputs(by pxm)
    size_t d_outputs_ = 1;                // dimension of outputs(by pxm)
    vector<float_type> mo_csr_val;
    vector<int> mo_csr_row_ptr;

    // csc variables: compressed sparse column format
    vector<float_type> csc_val;
    vector<int> csc_row_idx;
    vector<int> csc_col_ptr;

    // whether the dataset is to big: decide whether to use CPU when transferring csr to csc
    int use_cpu = false;
};

#endif //THUNDERGBM_DATASET_H
