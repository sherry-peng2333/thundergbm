//
// Created by zeyi on 1/9/19.
//

#include <thundergbm/trainer.h>
#include "thundergbm/parser.h"
#include "thundergbm/predictor.h"
#ifdef _WIN32
    INITIALIZE_EASYLOGGINGPP
#endif
int main(int argc, char **argv) {
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);

    GBMParam model_param;
    Parser parser;
    parser.parse_param(model_param, argc, argv);
    if(model_param.verbose == 0) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
    }
    else if (model_param.verbose == 1) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
    }

    if (!model_param.profiling) {
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
    }

    DataSet dataset;
    vector<vector<Tree>> boosted_model;

//    dataset.load_csc_from_file(model_param.path, model_param);
    dataset.load_from_file_mo(model_param.path, model_param);
    TreeTrainer trainer;
    boosted_model = trainer.train(model_param, dataset);
    parser.save_model("tgbm.model", model_param, boosted_model, dataset);

    // predict
    DataSet dataSet;
    model_param.path = "../../dataset/reg_valid_dataset.txt";
    dataSet.load_from_file_mo(model_param.path, model_param);
    Predictor pred;
    std::chrono::high_resolution_clock timer;
    auto start = timer.now();
    vector<float_type> y_pred_vec = pred.predict(model_param, boosted_model, dataSet);
    auto stop = timer.now();
    std::chrono::duration<float> valid_time = stop - start;
    LOG(INFO) << "valid time = " << valid_time.count();
    LOG(INFO) << "y_pred_vec: " << y_pred_vec;
}