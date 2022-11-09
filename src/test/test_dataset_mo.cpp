//
// Created by jiashuai on 17-9-17.
//
#include "gtest/gtest.h"
#include "thundergbm/dataset.h"
#include "omp.h"

class MODatasetTest : public ::testing::Test {
public:
    GBMParam param;
    vector<float_type> csr_val;
    vector<int> csr_row_ptr;
    vector<int> csr_col_idx;
    vector<float_type> y;
    size_t n_features_;
    vector<float_type> label;
    size_t d_outputs_ = 1;                // dimension of outputs(by pxm)
    vector<float_type> mo_csr_val;
    vector<int> mo_csr_row_ptr;
protected:
    void SetUp() override {
        param.depth = 6;
        param.n_trees = 40;
        param.n_device = 1;
        param.min_child_weight = 1;
        param.lambda = 1;
        param.gamma = 1;
        param.rt_eps = 1e-6;
        param.max_num_bin = 255;
        param.verbose = false;
        param.profiling = false;
        param.column_sampling_rate = 1;
        param.bagging = false;
        param.n_parallel_trees = 1;
        param.learning_rate = 1;
        param.objective = "mo-lab:mse";
        param.num_class = 1;
        param.path = "../../../dataset/test_dataset_mo.txt";
        param.tree_method = "auto";
        if (!param.verbose) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "True");
        }
        if (!param.profiling) {
            el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
        }
    }

    void load_from_file_mo(string file_name, GBMParam &param) {
        LOG(INFO) << "loading LIBSVM dataset from file \"" << file_name << "\"";
        // initialize
        csr_val.clear();
        csr_col_idx.clear();
        csr_row_ptr.resize(1, 0); // (0)
        mo_csr_val.clear();
        mo_csr_row_ptr.resize(1, 0); // (0)
        n_features_ = 0;
        d_outputs_ = 0;

        std::ifstream ifs(file_name, std::ifstream::binary);
        CHECK(ifs.is_open()) << "file " << file_name << " not found";

        int buffer_size = 4 << 20;
        char *buffer = (char *)malloc(buffer_size);
        //array may cause stack overflow in windows
        //std::array<char, 4> buffer{};
        const int nthread = omp_get_max_threads();

        auto find_last_line = [](char *ptr, const char *begin) {
            while (ptr != begin && *ptr != '\n' && *ptr != '\r' && *ptr != '\0') --ptr;
            return ptr;
        };

        while (ifs) {
            ifs.read(buffer, buffer_size);
            char *head = buffer;
            //ifs.read(buffer.data(), buffer.size());
            //char *head = buffer.data();
            size_t size = ifs.gcount();
            // create vectors for each thread
            vector<vector<float_type>> val_(nthread);
            vector<vector<int>> col_idx_(nthread);
            vector<vector<int>> row_len_(nthread);
            vector<vector<float_type>> mo_val_(nthread);
            vector<vector<int>> mo_row_len_(nthread);
            vector<int> max_feature(nthread, 0);
            vector<int> max_label(nthread, 0);
            bool fis_zero_base = false;                // feature indices start from 0
            bool lis_zero_base = true;                 // label indices start from 0

#pragma omp parallel num_threads(nthread)
            {
                //get working area of this thread
                int tid = omp_get_thread_num();
                size_t nstep = (size + nthread - 1) / nthread;
                size_t sbegin = (std::min)(tid * nstep, size - 1);
                size_t send = (std::min)((tid + 1) * nstep, size - 1);
                char *pbegin = find_last_line(head + sbegin, head);
                char *pend = find_last_line(head + send, head);

                //move stream start position to the end of last line
                if (tid == nthread - 1) {
                    if (ifs.eof())
                        pend = head + send;
                    else
                        ifs.seekg(-(head + send - pend), std::ios_base::cur);
                }

                //read instances line by line
                //TODO optimize parse line
                char *lbegin = pbegin;
                char *lend = lbegin;
                while (lend != pend) {
                    //get one line
                    lend = lbegin + 1;
                    while (lend != pend && *lend != '\n' && *lend != '\r' && *lend != '\0') {
                        ++lend;
                    }
                    string line(lbegin, lend);
                    if (line != "\n") {
                        std::stringstream ss(line);

                        //read label of an instance
                        float_type label;
                        mo_row_len_[tid].push_back(0);
                        string data, tmp;
                        ss >> data;
                        //std::cout << "data:" << data << std::endl;
                        std::stringstream input(data);
                        while(getline(input, tmp, ',')){
                            label = stof(tmp);
                            mo_val_[tid].push_back(label);
                            if(label>max_label[tid])
                                max_label[tid]=label;
                            mo_row_len_[tid].back()++;
                        }

                        row_len_[tid].push_back(0);
                        string tuple;
                        while (ss >> tuple) {
                            //std::cout << "tuple:" << tuple << std::endl;
                            int i;
                            float v;
                            CHECK_EQ(sscanf(tuple.c_str(), "%d:%f", &i, &v), 2)
                                << "read error, using [index]:[value] format";
//TODO one-based and zero-based
                            col_idx_[tid].push_back(i - 1);//one based
                            if(i - 1 == -1){
                                fis_zero_base = true;
                            }
                            CHECK_GE(i - 1, -1) << "dataset format error";
                            val_[tid].push_back(v);
                            if (i > max_feature[tid]) {
                                max_feature[tid] = i;
                            }
                            row_len_[tid].back()++;
                        }
                    }
                    //read next instance
                    lbegin = lend;

                }
            }
            for (int i = 0; i < nthread; i++) {
                if (max_feature[i] > n_features_)
                    n_features_ = max_feature[i];
            }
            // get the dimension of outputs
        if (param.objective.find("mo-lab:") != std::string::npos){ //multi-labels
            for (int i = 0; i < nthread; i++) {
                if (max_label[i] > d_outputs_)
                d_outputs_ = max_label[i];
            }
            if(lis_zero_base) d_outputs_=d_outputs_+1;
        }
        else if(param.objective.find("mo-reg:") != std::string::npos){ // multi-outputs regression
            d_outputs_=mo_row_len_[0][0];
        }
        // get the csr of features
        for (int tid = 0; tid < nthread; tid++) {
            csr_val.insert(csr_val.end(), val_[tid].begin(), val_[tid].end());
            if(fis_zero_base){
                for (int i = 0; i < col_idx_[tid].size(); ++i) {
                    col_idx_[tid][i]++;
                }
            }
            csr_col_idx.insert(csr_col_idx.end(), col_idx_[tid].begin(), col_idx_[tid].end());
            for (int row_len : row_len_[tid]) {
                csr_row_ptr.push_back(csr_row_ptr.back() + row_len);
            }
        }
        // get the csr of outputs
        for (int tid = 0; tid < nthread; tid++) {
            mo_csr_val.insert(mo_csr_val.end(), mo_val_[tid].begin(), mo_val_[tid].end());
            for (int mo_row_len : mo_row_len_[tid]) {
                mo_csr_row_ptr.push_back(mo_csr_row_ptr.back() + mo_row_len);
            }
        }
        }
        ifs.close();
        free(buffer);
    }
};

TEST_F(MODatasetTest, load_dataset_mo){
    DataSet dataset;
    load_from_file_mo(param.path, param);
    dataset.load_from_file_mo(param.path, param);
    printf("### Dataset: %s, num_instances: %ld, num_features: %ld, dim_outputs: %ld. ###\n",
           param.path.c_str(),
           dataset.n_instances_mo(),
           dataset.n_features(),
           dataset.d_outputs_);
    EXPECT_EQ(dataset.n_instances_mo(), 7395);
    EXPECT_EQ(dataset.n_features_, 1836);
    EXPECT_EQ(dataset.d_outputs_, 159);
    //EXPECT_EQ(dataset.label[0], -1);
    //EXPECT_EQ(dataset.csr_val[1], 1);

    for(int i = 0; i < csr_val.size(); i++)
        EXPECT_EQ(csr_val[i], dataset.csr_val[i]);
    for(int i = 0; i < csr_row_ptr.size(); i++)
        EXPECT_EQ(csr_row_ptr[i], dataset.csr_row_ptr[i]);
    for(int i = 0; i < csr_col_idx.size(); i++)
        EXPECT_EQ(csr_col_idx[i], dataset.csr_col_idx[i]);
    for(int i = 0; i < mo_csr_val.size(); i++)
        EXPECT_EQ(mo_csr_val[i], dataset.mo_csr_val[i]);
    for(int i = 0; i < mo_csr_row_ptr.size(); i++)
        EXPECT_EQ(mo_csr_row_ptr[i], dataset.mo_csr_row_ptr[i]);
}

