//
// Created by ss on 19-1-20.
//
#include "thundergbm/builder/hist_tree_builder.h"

#include "thundergbm/util/cub_wrapper.h"
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/discard_iterator.h"
#include "thrust/sequence.h"
#include "thrust/binary_search.h"
#include "thundergbm/util/multi_device.h"

void HistTreeBuilder::get_bin_ids() {
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        SparseColumns &columns = shards[device_id].columns;
        HistCut &cut = this->cut[device_id];
        auto &dense_bin_id = this->dense_bin_id[device_id];
        using namespace thrust;
        int n_column = columns.n_column;
        int nnz = columns.nnz;
        auto cut_row_ptr = cut.cut_row_ptr.device_data();
        auto cut_points_ptr = cut.cut_points_val.device_data();
        auto csc_val_data = columns.csc_val.device_data();
        SyncArray<unsigned char> bin_id;
        bin_id.resize(columns.nnz);
        auto bin_id_data = bin_id.device_data();
        int n_block = fminf((nnz / n_column - 1) / 256 + 1, 4 * 56);
        {
            auto lowerBound = [=]__device__(const float_type *search_begin, const float_type *search_end, float_type val) {
                const float_type *left = search_begin;
                const float_type *right = search_end - 1;

                while (left != right) {
                    const float_type *mid = left + (right - left) / 2;
                    if (*mid <= val)
                        right = mid;
                    else left = mid + 1;
                }
                return left;
            };
            TIMED_SCOPE(timerObj, "binning");
            device_loop_2d(n_column, columns.csc_col_ptr.device_data(), [=]__device__(int cid, int i) {
                auto search_begin = cut_points_ptr + cut_row_ptr[cid];
                auto search_end = cut_points_ptr + cut_row_ptr[cid + 1];
                auto val = csc_val_data[i];
                bin_id_data[i] = lowerBound(search_begin, search_end, val) - search_begin;
            }, n_block);
        }

        auto max_num_bin = param.max_num_bin;
        dense_bin_id.resize(n_instances * n_column);
        auto dense_bin_id_data = dense_bin_id.device_data();
        auto csc_row_idx_data = columns.csc_row_idx.device_data();
        device_loop(n_instances * n_column, [=]__device__(int i) {
        dense_bin_id_data[i] = max_num_bin;
    });
        device_loop_2d(n_column, columns.csc_col_ptr.device_data(), [=]__device__(int fid, int i) {
        int row = csc_row_idx_data[i];
        unsigned char bid = bin_id_data[i];
        dense_bin_id_data[row * n_column + fid] = bid;
    }, n_block);
    });
}

void HistTreeBuilder::find_split(int level, int device_id) {
    std::chrono::high_resolution_clock timer;

    const SparseColumns &columns = shards[device_id].columns;
    SyncArray<int> &nid = ins2node_id[device_id];
    SyncArray<GHPair> &gh_pair = gradients[device_id];
    Tree &tree = trees[device_id];
    SyncArray<SplitPoint> &sp = this->sp[device_id];
    SyncArray<bool> &ignored_set = shards[device_id].ignored_set;
    HistCut &cut = this->cut[device_id];
    auto &dense_bin_id = this->dense_bin_id[device_id];
    auto &last_hist = this->last_hist[device_id];

    TIMED_FUNC(timerObj);
    int n_nodes_in_level = static_cast<int>(pow(2, level));
    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = columns.n_column;
    int n_partition = n_column * n_nodes_in_level;
    int n_bins = cut.cut_points_val.size();
    int n_max_nodes = 2 << param.depth;
    int n_max_splits = n_max_nodes * n_bins;
    int n_split = n_nodes_in_level * n_bins;

    LOG(TRACE) << "start finding split";

    //find the best split locally
    {
        using namespace thrust;
        auto t_build_start = timer.now();

        //calculate split information for each split
        SyncArray<GHPair> hist(n_max_splits*d_outputs_);
        SyncArray<GHPair> missing_gh(n_partition*d_outputs_);
        auto cut_fid_data = cut.cut_fid.device_data();
        auto i2fid = [=] __device__(int i) { return cut_fid_data[i % n_bins]; };
        auto hist_fid = make_transform_iterator(counting_iterator<int>(0), i2fid);
        {
            {
                TIMED_SCOPE(timerObj, "build hist");
                {
                    size_t
                    smem_size = n_bins * sizeof(GHPair)*d_outputs_;
                    LOG(DEBUG) << "shared memory size = " << smem_size / 1024.0 << " KB";
                    if (n_nodes_in_level == 1) {
                        //root
                        auto hist_data = hist.device_data();
                        auto cut_row_ptr_data = cut.cut_row_ptr.device_data();
                        auto gh_data = gh_pair.device_data();
                        auto dense_bin_id_data = dense_bin_id.device_data();
                        auto max_num_bin = param.max_num_bin;
                        auto n_instances = this->n_instances;
                        if (smem_size > 600) {
                            //48 * 1024
                            device_loop(n_instances * n_column, [=]__device__(int i) {
                                int iid = i / n_column;
                                int fid = i % n_column;
                                unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                                if (bid != max_num_bin) {
                                    int feature_offset = cut_row_ptr_data[fid];
                                    for(int i = 0; i < d_outputs_; i++){
                                        const GHPair src = gh_data[iid*d_outputs_+i];
                                        GHPair &dest = hist_data[i*n_max_splits+feature_offset + bid];
                                        if(src.h != 0)
                                            atomicAdd(&dest.h, src.h);
                                        if(src.g != 0)
                                            atomicAdd(&dest.g, src.g);
                                    }

                                }
                            });
                        } else {
                            // To do: add multi-outputs feature 
                            int num_fv = n_instances * n_column;
                            anonymous_kernel([=]__device__() {
                                extern __shared__ GHPair local_hist[];
                                for (int i = threadIdx.x; i < n_bins; i += blockDim.x) {
                                    local_hist[i] = 0;
                                }
                                __syncthreads();
                                for (int i = blockIdx.x * blockDim.x + threadIdx.x;
                                     i < num_fv; i += blockDim.x * gridDim.x) {
                                    int iid = i / n_column;
                                    int fid = i % n_column;
                                    unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                                    if (bid != max_num_bin) {
                                        int feature_offset = cut_row_ptr_data[fid];
                                        const GHPair src = gh_data[iid];
                                        GHPair &dest = local_hist[feature_offset + bid];
                                        if(src.h != 0)
                                            atomicAdd(&dest.h, src.h);
                                        if(src.g != 0)
                                            atomicAdd(&dest.g, src.g);

                                    }
                                }
                                __syncthreads();
                                for (int i = threadIdx.x; i < n_bins; i += blockDim.x) {
                                    GHPair &dest = hist_data[i];
                                    GHPair src = local_hist[i];
                                    if(src.h != 0)
                                        atomicAdd(&dest.h, src.h);
                                    if(src.g != 0)
                                        atomicAdd(&dest.g, src.g);

                                }
                            }, num_fv, smem_size);
                        }
                    } else {
                        //otherwise
                        auto t_dp_begin = timer.now();
                        SyncArray<int> node_idx(n_instances);
                        SyncArray<int> node_ptr(n_nodes_in_level + 1);
                        {
                            TIMED_SCOPE(timerObj, "data partitioning");
                            SyncArray<int> nid4sort(n_instances);
                            nid4sort.copy_from(ins2node_id[device_id]);
                            sequence(cuda::par, node_idx.device_data(), node_idx.device_end(), 0);
                            cub_sort_by_key(nid4sort, node_idx);
                            auto counting_iter = make_counting_iterator < int > (nid_offset);
                            node_ptr.host_data()[0] =
                                    lower_bound(cuda::par, nid4sort.device_data(), nid4sort.device_end(), nid_offset) -
                                    nid4sort.device_data();

                            upper_bound(cuda::par, nid4sort.device_data(), nid4sort.device_end(), counting_iter,
                                        counting_iter + n_nodes_in_level, node_ptr.device_data() + 1);
                            LOG(DEBUG) << "node ptr = " << node_ptr;
                            cudaDeviceSynchronize();
                        }
                        auto t_dp_end = timer.now();
                        std::chrono::duration<double> dp_used_time = t_dp_end - t_dp_begin;
                        this->total_dp_time += dp_used_time.count();


                        auto node_ptr_data = node_ptr.host_data();
                        auto node_idx_data = node_idx.device_data();
                        auto cut_row_ptr_data = cut.cut_row_ptr.device_data();
                        auto gh_data = gh_pair.device_data();
                        auto dense_bin_id_data = dense_bin_id.device_data();
                        auto max_num_bin = param.max_num_bin;
                        for (int i = 0; i < n_nodes_in_level / 2; ++i) {

                            int nid0_to_compute = i * 2;
                            int nid0_to_substract = i * 2 + 1;
                            int n_ins_left = node_ptr_data[nid0_to_compute + 1] - node_ptr_data[nid0_to_compute];
                            int n_ins_right = node_ptr_data[nid0_to_substract + 1] - node_ptr_data[nid0_to_substract];
                            if (max(n_ins_left, n_ins_right) == 0) continue;
                            if (n_ins_left > n_ins_right)
                                swap(nid0_to_compute, nid0_to_substract);

                            //compute
                            {
                                int nid0 = nid0_to_compute;
                                auto idx_begin = node_ptr.host_data()[nid0];
                                auto idx_end = node_ptr.host_data()[nid0 + 1];
                                auto hist_data = hist.device_data();
                                //auto hist_data = hist.device_data() + nid0 * n_bins * d_outputs_;
                                this->total_hist_num++;

                                if (smem_size > 600) {
                                    //48 * 1024
                                    device_loop((idx_end - idx_begin) * n_column, [=]__device__(int i) {
                                        int iid = node_idx_data[i / n_column + idx_begin];
                                        int fid = i % n_column;
                                        unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                                        if (bid != max_num_bin) {
                                            int feature_offset = cut_row_ptr_data[fid];
                                            for(int i = 0; i < d_outputs_; i++){
                                                const GHPair src = gh_data[iid*d_outputs_+i];
                                                GHPair &dest = hist_data[i*n_max_splits+nid0*n_bins+feature_offset + bid];
                                                if(src.h != 0)
                                                    atomicAdd(&dest.h, src.h);
                                                if(src.g != 0)
                                                    atomicAdd(&dest.g, src.g);
                                            }
                                        }
                                    });
                                } else {
                                    // To do: add multi-outputs feature
                                    int num_fv = (idx_end - idx_begin) * n_column;
                                    anonymous_kernel([=] __device__() {
                                        extern __shared__ GHPair local_hist[];
                                        for (int i = threadIdx.x; i < n_bins; i += blockDim.x) {
                                            local_hist[i] = 0;
                                        }
                                        __syncthreads();

                                        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
                                             i < num_fv; i += blockDim.x * gridDim.x) {
                                            int iid = node_idx_data[i / n_column + idx_begin];
                                            //int fid = i - n_column *( i / n_column);
                                            int fid = i % n_column;
                                            unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                                            if (bid != max_num_bin) {
                                                const GHPair src = gh_data[iid];
                                                int feature_offset = cut_row_ptr_data[fid];
                                                GHPair &dest = local_hist[feature_offset + bid];
                                                if(src.h != 0)
                                                    atomicAdd(&dest.h, src.h);
                                                if(src.g != 0)
                                                    atomicAdd(&dest.g, src.g);
                                            }
                                        }

                                        __syncthreads();
                                        for (int i = threadIdx.x; i < n_bins; i += blockDim.x) {
                                            GHPair src = local_hist[i];
                                            GHPair &dest = hist_data[i];
                                            if(src.h != 0)
                                                atomicAdd(&dest.h, src.h);
                                            if(src.g != 0)
                                                atomicAdd(&dest.g, src.g);
                                        }
                                    }, num_fv, smem_size);
                                }
                            }

                            //subtract
                            auto t_copy_start = timer.now();
                            {
                                auto hist_data_computed = hist.device_data() + nid0_to_compute * n_bins;
                                auto hist_data_to_compute = hist.device_data() + nid0_to_substract * n_bins;
                                auto father_hist_data = last_hist.device_data() + (nid0_to_substract / 2) * n_bins ;
                                //auto hist_data = hist.device_data();
                                //auto father_hist_data = last_hist.device_data();
                                device_loop(n_bins * d_outputs_, [=]__device__(int i) {
                                    int d = i / n_bins;
                                    int bid = i % n_bins;
                                    hist_data_computed[d*n_max_splits+bid] = father_hist_data[d*n_max_splits+bid]-hist_data_computed[d*n_max_splits+bid];
                                });
                            }
                            auto t_copy_end = timer.now();
                            std::chrono::duration<double> cp_used_time = t_copy_end - t_copy_start;
                            this->total_copy_time += cp_used_time.count();
//                            PERFORMANCE_CHECKPOINT(timerObj);
                        }  // end for each node
                    }//end # node > 1
                    last_hist.copy_from(hist);
                    cudaDeviceSynchronize();
                }
                LOG(DEBUG) << "level: " << level;
                LOG(DEBUG) << "hist new = " << hist;
                auto t_build_hist_end = timer.now();
                std::chrono::duration<double> bh_used_time = t_build_hist_end - t_build_start;
                this->build_hist_used_time += bh_used_time.count();
                this->build_n_hist++;
                LOG(DEBUG) << "-------------->>> build_hist_used_time: " << bh_used_time.count();
                LOG(DEBUG) << "-------------->>> build_num_hist: " << this->build_n_hist;
                LOG(DEBUG) << "-------------->>> total_build_hist_used_time: " << this->build_hist_used_time - this->total_dp_time;
                LOG(DEBUG) << "-------------->>> n_hist::::: " << this->total_hist_num;
                LOG(DEBUG) << "-------------->>> dp_time::::: " << this->total_dp_time;
                LOG(DEBUG) << "-------------->>> cp_time::::: " << this->total_copy_time;

                //LOG(DEBUG) << "cutfid = " << cut.cut_fid;
                inclusive_scan_by_key(cuda::par, hist_fid, hist_fid + n_split*d_outputs_,
                                      hist.device_data(), hist.device_data());
                LOG(DEBUG) << hist;

                auto nodes_data = tree.nodes.device_data();
                auto missing_gh_data = missing_gh.device_data();
                auto cut_row_ptr = cut.cut_row_ptr.device_data();
                auto hist_data = hist.device_data();

                for(int pid = 0; pid < n_partition; pid++){
                    int nid0 = pid / n_column;
                    int nid = nid0 + nid_offset;
                    LOG(INFO) << "checkpoint ...";
                    if (!nodes_data[nid].splittable()) return;
                    int fid = pid % n_column;
                    auto sum_gh_pair_data = nodes_data[nid].sum_gh_pair.device_data();
                    if (cut_row_ptr[fid + 1] != cut_row_ptr[fid]){
                        device_loop(d_outputs_, [=]__device__(int i){
                            GHPair node_gh = hist_data[i*n_partition+nid0 * n_bins+cut_row_ptr[fid + 1] - 1];
                            missing_gh_data[i*n_partition+pid] = sum_gh_pair_data[i] - node_gh;
                        });
                    }
                }
                LOG(DEBUG) << missing_gh;
            }
        }
        //calculate gain of each split
        SyncArray<float_type> gain(n_max_splits);
        {
//            TIMED_SCOPE(timerObj, "calculate gain");
            auto compute_gain = [](GHPair father, GHPair lch, GHPair rch, float_type min_child_weight, float_type lambda) -> float {
                    if (lch.h >= min_child_weight && rch.h >= min_child_weight)
                    return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
            (father.g * father.g) / (father.h + lambda);
                    else
                    return 0;
            };

            const Tree::TreeNode *nodes_data = tree.nodes.host_data();
            GHPair *gh_prefix_sum_data = hist.host_data();
            float_type *gain_data = gain.host_data();
            const auto missing_gh_data = missing_gh.host_data();
            auto ignored_set_data = ignored_set.host_data();
            //for lambda expression
            float_type mcw = param.min_child_weight;
            float_type l = param.lambda;

            for(int i = 0; i < n_split; i++){
                int nid0 = i / n_bins;
                int nid = nid0 + nid_offset;
                int fid = hist_fid[i % n_bins];
                if (nodes_data[nid].is_valid && !ignored_set_data[fid]) {
                    int pid = nid0 * n_column + hist_fid[i];
                    auto father_gh = nodes_data[nid].sum_gh_pair.host_data();
                    float_type default_to_left_gain = 0;
                    float_type default_to_right_gain = 0;
                    for(int j = 0; j < d_outputs_; j++){
                        GHPair rch_gh = gh_prefix_sum_data[j*n_max_splits+i];
                        default_to_left_gain = default_to_left_gain + max(0.f,
                            compute_gain(father_gh[j], father_gh[j] - rch_gh, rch_gh, mcw, l));
                    }
                    for(int j = 0; j < d_outputs_; j++){
                        GHPair rch_gh = gh_prefix_sum_data[j*n_max_splits+i]+missing_gh_data[j*n_partition+pid];
                        default_to_right_gain = max(0.f,
                            compute_gain(father_gh[j], father_gh[j] - rch_gh, rch_gh, mcw, l));
                    }
                    if (default_to_left_gain > default_to_right_gain)
                        gain_data[i] = default_to_left_gain;
                    else
                        gain_data[i] = -default_to_right_gain;//negative means default split to right

                } else gain_data[i] = 0;
            }

            LOG(DEBUG) << "gain = " << gain;
        }

        SyncArray<int_float> best_idx_gain(n_nodes_in_level);
        {
//            TIMED_SCOPE(timerObj, "get best gain");
            auto arg_abs_max = []__device__(const int_float &a, const int_float &b) {
                if (fabsf(get<1>(a)) == fabsf(get<1>(b)))
                    return get<0>(a) < get<0>(b) ? a : b;
                else
                    return fabsf(get<1>(a)) > fabsf(get<1>(b)) ? a : b;
            };

            auto nid_iterator = make_transform_iterator(counting_iterator<int>(0), placeholders::_1 / n_bins);

            reduce_by_key(
                    cuda::par,
                    nid_iterator, nid_iterator + n_split,
                    make_zip_iterator(make_tuple(counting_iterator<int>(0), gain.device_data())),
                    make_discard_iterator(),
                    best_idx_gain.device_data(),
                    thrust::equal_to<int>(),
                    arg_abs_max
            );
            LOG(DEBUG) << n_split;
            LOG(DEBUG) << "best rank & gain = " << best_idx_gain;
        }

        //get split points
        {
            const int_float *best_idx_gain_data = best_idx_gain.device_data();
            auto hist_data = hist.device_data();
            const auto missing_gh_data = missing_gh.device_data();
            auto cut_val_data = cut.cut_points_val.device_data();

            sp.resize(n_nodes_in_level);
            auto sp_data = sp.device_data();
            auto nodes_data = tree.nodes.device_data();

            int column_offset = columns.column_offset;

            auto cut_row_ptr_data = cut.cut_row_ptr.device_data();
            device_loop(n_nodes_in_level, [=]__device__(int i) {
                int_float bst = best_idx_gain_data[i];
                float_type best_split_gain = get<1>(bst);
                int split_index = get<0>(bst);
                if (!nodes_data[i + nid_offset].is_valid) {
                    sp_data[i].split_fea_id = -1;
                    sp_data[i].nid = -1;
                    return;
                }
                int fid = hist_fid[split_index];
                sp_data[i].split_fea_id = fid + column_offset;
                sp_data[i].nid = i + nid_offset;
                sp_data[i].gain = fabsf(best_split_gain);
                sp_data[i].fval = cut_val_data[split_index % n_bins];
                sp_data[i].split_bid = (unsigned char) (split_index % n_bins - cut_row_ptr_data[fid]);
                sp_data[i].default_right = best_split_gain < 0;
            });

            for(int i = 0; i < n_nodes_in_level; i++){
                int_float bst = best_idx_gain_data[i];
                float_type best_split_gain = get<1>(bst);
                int split_index = get<0>(bst);
                auto spi_fea_missing_gh_data = sp_data[i].fea_missing_gh.device_data();
                auto spi_rch_sum_gh_data = sp_data[i].rch_sum_gh.device_data();
                device_loop(d_outputs_, [=]__device__(int j){
                    spi_fea_missing_gh_data[j] = missing_gh_data[j * n_partition + i * n_column + hist_fid[split_index]];
                    spi_rch_sum_gh_data[j] = hist_data[j * n_max_splits + split_index];
                }
                );
            }

        }
    }

    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}

void HistTreeBuilder::update_ins2node_id() {
    TIMED_FUNC(timerObj);
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        SyncArray<bool> has_splittable(1);
        auto &columns = shards[device_id].columns;
        //set new node id for each instance
        {
//        TIMED_SCOPE(timerObj, "get new node id");
            auto nid_data = ins2node_id[device_id].device_data();
            const Tree::TreeNode *nodes_data = trees[device_id].nodes.device_data();
            has_splittable.host_data()[0] = false;
            bool *h_s_data = has_splittable.device_data();
            int column_offset = columns.column_offset;

            int n_column = columns.n_column;
            auto dense_bin_id_data = dense_bin_id[device_id].device_data();
            int max_num_bin = param.max_num_bin;
            device_loop(n_instances, [=]__device__(int iid) {
                int nid = nid_data[iid];
                const Tree::TreeNode &node = nodes_data[nid];
                int split_fid = node.split_feature_id;
                if (node.splittable() && ((split_fid - column_offset < n_column) && (split_fid >= column_offset))) {
                    h_s_data[0] = true;
                    unsigned char split_bid = node.split_bid;
                    unsigned char bid = dense_bin_id_data[iid * n_column + split_fid - column_offset];
                    bool to_left = true;
                    if ((bid == max_num_bin && node.default_right) || (bid <= split_bid))
                        to_left = false;
                    if (to_left) {
                        //goes to left child
                        nid_data[iid] = node.lch_index;
                    } else {
                        //right child
                        nid_data[iid] = node.rch_index;
                    }
                }
            });
        }
        LOG(DEBUG) << "new tree_id = " << ins2node_id[device_id];
        has_split[device_id] = has_splittable.host_data()[0];
    });
}

void HistTreeBuilder::init(const DataSet &dataset, const GBMParam &param) {
    TreeBuilder::init(dataset, param);
    //TODO refactor
    //init shards
    int n_device = param.n_device;
    shards = vector<Shard>(n_device);
    vector<std::unique_ptr<SparseColumns>> v_columns(param.n_device);
    for (int i = 0; i < param.n_device; ++i) {
        v_columns[i].reset(&shards[i].columns);
        shards[i].ignored_set = SyncArray<bool>(dataset.n_features());
    }
    SparseColumns columns;
    if(dataset.use_cpu)
        columns.csr2csc_cpu(dataset, v_columns);
    else
        columns.csr2csc_gpu(dataset, v_columns);
    cut = vector<HistCut>(param.n_device);
    dense_bin_id = MSyncArray<unsigned char>(param.n_device);
    last_hist = MSyncArray<GHPair>(param.n_device);
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        if(dataset.use_cpu)
            cut[device_id].get_cut_points2(shards[device_id].columns, param.max_num_bin, n_instances);
        else
            cut[device_id].get_cut_points3(shards[device_id].columns, param.max_num_bin, n_instances);
        last_hist[device_id].resize((2 << param.depth) * cut[device_id].cut_points_val.size()* dataset.d_outputs_);
    });
    get_bin_ids();
    for (int i = 0; i < param.n_device; ++i) {
        v_columns[i].release();
    }
    // SyncMem::clear_cache();
    int gpu_num;
    cudaError_t err = cudaGetDeviceCount(&gpu_num);
    std::atexit([](){
        SyncMem::clear_cache();
    });
}
