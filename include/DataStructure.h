//
// Created by Josh Briegal on 23/11/2017.
//

#ifndef C_DATASTRUCTURE_H
#define C_DATASTRUCTURE_H

#include <vector>

class DataStructure{
    /*
     * Data and data-manipulation functions stored here. Can be initialised with or without X errors and from a file
     * containing this data in the format t, X, (X_error).
     */
    std::vector<double> X; std::vector<double> t; std::vector<double> X_err;
    double X_mean; double t_median; double t_max; double t_length;
    std::vector<double> norm_t; std::vector<double> norm_X;

    void setXMean();
    void settMedian();
    void settMax();
    void calcNormt();
    void calcNormX();

public:
    explicit DataStructure(const std::string &filename);
    DataStructure(std::vector<double>*, std::vector<double>*);
    DataStructure(std::vector<double>*, std::vector<double>*, std::vector<double>*);

    // returnable values from the dataset
    std::vector<double>* values();
    std::vector<double>* errors();
    std::vector<double>* timeseries();
    std::vector<double>* normalised_timeseries();
    std::vector<double>* normalised_values();
    double mean_X();
    double median_time() ;
    double max_time();
};

#endif //C_DATASTRUCTURE_H