//
// Created by Josh Briegal on 23/11/2017.
//

#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "DataStructure.h"

const char* EmptyDataStructureException::what() const throw(){
    return "DataStructure cannot be initialised with empty constructor";
}

EmptyDataStructureException EmptyDSEx;

const char* BadDataFileReadException::what() const throw(){
    return "DataStructure cannot be initialised from file";
}

BadDataFileReadException BadFileEx;

DataStructure::DataStructure(std::vector<double>* t_in, std::vector<double>* X_in){
    if(X_in->empty() || t_in->empty()){
        throw EmptyDSEx;
    }
    X = *X_in;
    t = *t_in;
    setXMean(); settMax();
    norm_t.resize(t.size());
    norm_X.resize(X.size());
    calcNormt(); calcNormX();
    settMedian();
};

DataStructure::DataStructure(std::vector<double>* t_in, std::vector<double>* X_in, std::vector<double>* X_err_in):
        DataStructure(X_in,t_in){
    X_err = *X_err_in;
};

DataStructure::DataStructure(const std::string &filename){
    /*
     * constructor for DataStructure from file
     */
    bool has_errors = 1;
    bool update_has_errors = 1;
    std::ifstream file(filename, std::ios::in);
    if(file.fail()){
        throw BadFileEx;
    }
    std::string line;
    std::vector<double> X_in; std::vector<double> t_in; std::vector<double> X_err_in;
    if(file.good()){
        while (std::getline(file, line, '\n')) {
            if(line[0] != '#') {  // ignore commented lines
                double X_val;
                double t_val;
                double X_err_val;

                if(line.find_first_of(",") != std::string::npos){
                    std::replace(line.begin(), line.end(), ',', ' ');  // replace commas with spaces
                }

                std::stringstream linestream(line);
                while(linestream >> t_val >> X_val){
                    t_in.push_back(t_val);
                    X_in.push_back(X_val);
                    while(linestream >> X_err_val && has_errors){
                        X_err_in.push_back(X_err_val);
                        update_has_errors = 0;
                    }
                    if(update_has_errors){
                        has_errors = 0;
                    }
                }
            }
        }
    }
    if(has_errors){
        *this = DataStructure(&t_in, &X_in, &X_err_in);
    } else {
        *this = DataStructure(&t_in, &X_in);
    }
};

void DataStructure::setXMean(){  // ignoring any 'NaN' values
    std::vector<double> x_copy (sizeof(X));
    auto const end = std::remove_copy_if(X.begin(), X.end(), x_copy.begin(), std::isnan<double>);
    std::cout << std::endl;
    X_mean = accumulate(x_copy.begin(), end, 0.0)/std::distance(x_copy.begin(), end);
}

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

std::vector<double>* DataStructure::rvalues() { return &X; };
std::vector<double>* DataStructure::rerrors(){ return &X_err; };
std::vector<double>* DataStructure::rtimeseries(){ return &t; };
std::vector<double>* DataStructure::rnormalised_timeseries(){ return &norm_t; };
std::vector<double>* DataStructure::rnormalised_values(){ return &norm_X; };

std::vector<double> DataStructure::values() { return X; };
std::vector<double> DataStructure::errors(){ return X_err; };
std::vector<double> DataStructure::timeseries(){ return t; };
std::vector<double> DataStructure::normalised_timeseries(){ return norm_t; };
std::vector<double> DataStructure::normalised_values(){ return norm_X; };



double DataStructure::mean_X() { return X_mean; };
double DataStructure::median_time() { return t_median; };
double DataStructure::max_time() { return t_max; };

