//
// Created by ss on 19-1-2.
//

#ifndef THUNDERGBM_SHARD_H
#define THUNDERGBM_SHARD_H


#include "thundergbm/sparse_columns.h"
#include "thundergbm/tree.h"

class SplitPoint;

struct Shard {
    SparseColumns columns;//a subset of columns (or features)
    SyncArray<bool> ignored_set;//for column sampling

    void column_sampling(float rate);
};



class SplitPoint {
public:
    float_type gain;
    SyncArray<GHPair> fea_missing_gh;//missing gh in this segment
    SyncArray<GHPair> rch_sum_gh;//right child total gh (missing gh included if default2right)
    bool default_right;
    int nid;

    //split condition
    int split_fea_id;
    float_type fval;//split on this feature value (for exact)
    unsigned char split_bid;//split on this bin id (for hist)

    SplitPoint() {
        nid = -1;
        split_fea_id = -1;
        gain = 0;
    }

    friend std::ostream &operator<<(std::ostream &output, const SplitPoint &sp) {
        output << sp.gain << "/" << sp.split_fea_id << "/" << sp.nid << "/" << sp.rch_sum_gh;
        return output;
    }
};

#endif //THUNDERGBM_SHARD_H
