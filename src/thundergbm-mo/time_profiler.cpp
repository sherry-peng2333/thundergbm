////
//// Created by xuemei on 23-3-10.
////
//#include "time_profiler.h"
//#include "chrono"
//
//TIMETYPE TimeProfiler::time_starts;
//TIMETYPE TimeProfiler::time_ends;
//map<string, double> TimeProfiler::time_summations;
//map<string, int> TimeProfiler::counter;
//
//double time_diff(time_point<high_resolution_clock> &t_start, time_point<high_resolution_clock> &t_end) {
//    duration<double> diff = t_end - t_start;
//    return diff.count();
//}
//
//void TimeProfiler::timer_start(string code_frag_name){
//    time_starts[code_frag_name] = timer.now();
//}
//
//void TimeProfiler::timer_end(string code_frag_name){
//    time_ends[code_frag_name] = timer.now();
//    map<string, double>::iterator it = time_summations.find(code_frag_name);
//    if(it == time_summations.end()) {
//        time_summations[code_frag_name] = 0;
//        counter[code_frag_name] = 0;
//    }
//    time_summations[code_frag_name] += time_diff(time_starts[code_frag_name], time_ends[code_frag_name]);
//    counter[code_frag_name]++;
//}
//
//void TimeProfiler::profile_report(){
//    std::cout << "TimeProfiler: \n";
//    map<string, double>::iterator iter = time_summations.begin();
//    while(iter != time_summations.end()) {
//        std::cout << "  >>> " << iter->first << ": " << iter->second << "\n";
//        iter++;
//    }
//}
//
//void TimeProfiler::profile_report(string code_frag_name) {
//    map<string, double>::iterator iter = time_summations.find(code_frag_name);
//    if(iter == time_summations.end())
//        std::cout << "no code fragment: " << code_frag_name;
//    else
//        std::cout << " >>> " << iter->first << ": " << iter->second << "\n";
//}