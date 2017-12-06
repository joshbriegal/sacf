//
// Created by Josh Briegal on 23/11/2017.
//

#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <algorithm>

#include "DataStructure.h"

DataStructure::DataStructure(std::vector<double>* X_in, std::vector<double>* t_in){
    X = *X_in;
    t = *t_in;
    setXMean(); settMax();
    norm_t.resize(t.size());
    norm_X.resize(X.size());
    calcNormt(); calcNormX();
    settMedian();
};

DataStructure::DataStructure(std::vector<double>* X_in, std::vector<double>* t_in, std::vector<double>* X_err_in):
        DataStructure(X_in,t_in){
    X_err = *X_err_in;
};

DataStructure::DataStructure(const std::string &filename){
    /*
     * constructor for DataStructure from file
     */
    std::ifstream file(filename, std::ios::in);
    std::string line;
    std::vector<double> X_in; std::vector<double> t_in; std::vector<double> X_err_in;
    if(file.good()){
        while (getline(file, line)) {
            if(line[0] != '#') {
                std::stringstream linestream(line);
                std::string data;
                double X_val;
                double t_val;
                double X_err_val;

                linestream >> t_val >> X_val >> X_err_val;
                t_in.push_back(t_val);
                X_in.push_back(X_val);
                X_err_in.push_back(X_err_val);
            }
        }
    }
    *this = DataStructure(&X_in, &t_in, &X_err_in);
};

void DataStructure::setXMean(){X_mean = accumulate(X.begin(), X.end(), 0.0)/X.size();}

void DataStructure::settMax(){ t_max = t.back(); } // as time array is ordered

void DataStructure::calcNormt(){
    transform(t.begin(), t.end(), norm_t.begin(), std::bind2nd(std::minus<double>(), t[0]));
};

void DataStructure::calcNormX(){
    transform(X.begin(), X.end(), norm_X.begin(), std::bind2nd(std::minus<double>(), X_mean));
};

void DataStructure::settMedian(){ //with respect to normalised timeseries
    size_t size = norm_t.size();
    if(size  % 2 == 0) {t_median = (norm_t[size / 2 - 1] + norm_t[size / 2]) / 2;}
    else {t_median = norm_t[size / 2];}
}

std::vector<double>* DataStructure::values() { return &X; };
std::vector<double>* DataStructure::errors(){ return &X_err; };
std::vector<double>* DataStructure::timeseries(){ return &t; };
std::vector<double>* DataStructure::normalised_timeseries(){ return &norm_t; };
std::vector<double>* DataStructure::normalised_values(){ return &norm_X; };

double DataStructure::mean_X() { return X_mean; };
double DataStructure::median_time() { return t_median; };
double DataStructure::max_time() { return t_max; };

