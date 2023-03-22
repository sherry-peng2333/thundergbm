////
//// Created by xuemei on 23-3-10.
////
//
//#ifndef THUNDERGBM_TIME_PROFILER_H
//#define THUNDERGBM_TIME_PROFILER_H
//#include "iostream"
//#include "map"
//#include "sys/types.h"
//#include "sys/stat.h"
//#include "fcntl.h"
//#include "common.h"
//using namespace std::chrono;
//using namespace std;
//typedef map<string, time_point<high_resolution_clock>> TIMETYPE;
//
//class TimeProfiler{
//public:
//    static map<string, double> time_summations;
//    static TIMETYPE time_starts;
//    static TIMETYPE time_ends;
//    //map<string, time_point<high_resolution_clock>> time_starts;
//    //map<string, time_point<high_resolution_clock>> time_ends;
//    //map<string, double> time_summations_asm;
//    //map<string, time_point<high_resolution_clock>> time_start_asm;
//    //map<string, time_point<high_resolution_clock>> time_end_asm;
//    static map<string, int> counter;
//    static t_clock timer;
//
//    //
//    static void timer_start(string code_frag_name);
//    static void timer_end(string code_frag_name);
//    static void profile_report();
//    static void profile_report(string code_frag_name);
//    void timer_start_asm(string code_frag_name);
//    void timer_end_asm(string code_frag_name);
//    void profile_report_asm();
//    void profile_report_asm(string code_frag_name);
//    template<typename T>
//    T vector_variance(T *vector, T *vector_compare, int num_items) {
//        T variance = 0;
//        T tmp;
//        for(int i = 0; i < num_items; i++) {
//            tmp = vector_compare[i] - vector[i];
//            variance += tmp * tmp;
//        }
//
//        return variance;
//    }
//
//};
//#endif //THUNDERGBM_TIME_PROFILER_H
