//
// Created by Josh Briegal on 23/11/2017.
//

#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <functional>

#include "sacf/DataStructure.h"


const char* EmptyDataStructureException::what() const throw(){
    return "DataStructure cannot be initialised with empty constructor";
}

EmptyDataStructureException EmptyDSEx;

const char* BadDataFileReadException::what() const throw(){
    return "DataStructure cannot be initialised from file";
}

BadDataFileReadException BadFileEx;

const char* BadDataInputException::what() const throw(){
    return "DataStructure cannot be initialised with inputs - check lists have the same number of datapoints";
}

BadDataInputException BadDataEx;

void DataStructure::setUp(){
    if (_data.empty() || t.empty()){ throw EmptyDSEx; }
    if (_data.size() % t.size() != 0){ throw BadDataEx; }
    setDataMean();
    settMax();
    norm_t.resize(t.size());
    norm_data.resize(_data.size());
    calcNormt(); calcNormData();
    settMedian();
}

long DataStructure::getVectorIndex(int i, int j){
    return (long) i + j * M_datapoints;
}

DataStructure::DataStructure(std::vector<double>* t_in, std::vector< std::vector<double> >* data_in): t(*t_in){
    for(auto const& data_series: *data_in){
        for(auto const X: data_series){
            _data.push_back(X);
        }
    }
    N_datasets = data_in->size();
    M_datapoints = t_in->size();
    setUp();
};

DataStructure::DataStructure(std::vector<double>* t_in, std::vector< std::vector<double> >* data_in,
                             std::vector< std::vector<double> >* data_err_in): t(*t_in){
    for(auto const& data_series: *data_in){
        for(auto const X: data_series){
            _data.push_back(X);
        }
    }
    for(auto const& data_series: *data_err_in){
        for(auto const X: data_series){
            err.push_back(X);
        }
    }
    N_datasets = data_in->size();
    M_datapoints = t_in->size();
    setUp();
};

DataStructure::DataStructure(std::vector<double>* t_in, std::vector<double>* data_in):
                             _data(*data_in), t(*t_in){
    N_datasets = 1;
    M_datapoints = t_in->size();
    setUp();
};

DataStructure::DataStructure(std::vector<double>* t_in, std::vector<double>* data_in, std::vector<double>* data_err_in):
                             _data(*data_in), t(*t_in), err(*data_err_in){
    N_datasets = 1;
    M_datapoints = t_in->size();
    setUp();
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
    std::vector<double> data_in = std::vector<double>();
    std::vector<double> t_in = std::vector<double>();
    std::vector<double> data_err_in = std::vector<double>();
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
                    data_in.push_back(X_val);
                    while(linestream >> X_err_val && has_errors){
                        data_err_in.push_back(X_err_val);
                        update_has_errors = 0;
                    }
                    if(update_has_errors){
                        has_errors = 0;
                    }
                }
            }
        }
    }
    _data = data_in;
    t = t_in;
    if(has_errors){
        err = data_err_in;
    }
    N_datasets = 1;
    M_datapoints = t_in.size();
    setUp();
};

void DataStructure::setDataMean(){  // ignoring any 'NaN' values
    data_mean = std::vector<double>(N_datasets);
    for(int j = 0; j < N_datasets; j++){
        std::vector<double> x_copy(M_datapoints);
        auto const end = std::remove_copy_if(_data.begin() + getVectorIndex(0, j),
                                             _data.begin() + getVectorIndex(M_datapoints, j),
                                             x_copy.begin(), [](double d) { return std::isnan(d); });
        data_mean[j] = accumulate(x_copy.begin(), end, 0.0)/std::distance(x_copy.begin(), end);
    }
}

void DataStructure::settMax(){ t_max = t.back(); } // as time array is ordered

void DataStructure::calcNormt(){
    transform(t.begin(), t.end(), norm_t.begin(), std::bind(std::minus<double>(), std::placeholders::_1, t[0]));
};

void DataStructure::calcNormData(){
    norm_data = std::vector<double>(_data.size());
    for(int j = 0; j<N_datasets; j++){
        transform(_data.begin() + getVectorIndex(0, j), _data.begin() + getVectorIndex(M_datapoints, j),
         norm_data.begin() + getVectorIndex(0, j), std::bind(std::minus<double>(), std::placeholders::_1, data_mean[j]));
    }
};

void DataStructure::settMedian(){ //with respect to normalised timeseries
    size_t size = norm_t.size();
    if(size  % 2 == 0) {t_median = (norm_t[size / 2 - 1] + norm_t[size / 2]) / 2;}
    else {t_median = norm_t[size / 2];}
}

std::vector<double>* DataStructure::rdata() { return &_data; };
std::vector<double>* DataStructure::rerrors(){ return &err; };
std::vector<double>* DataStructure::rtimeseries(){ return &t; };
std::vector<double>* DataStructure::rnormalised_timeseries(){ return &norm_t; };
std::vector<double>* DataStructure::rnormalised_data(){ return &norm_data; };

std::vector<double> DataStructure::data() { return _data; };
std::vector<double> DataStructure::errors(){ return err; };
std::vector<double> DataStructure::timeseries(){ return t; };
std::vector<double> DataStructure::normalised_timeseries(){ return norm_t; };
std::vector<double> DataStructure::normalised_data(){ return norm_data; };


std::vector<double> DataStructure::mean_data() { return data_mean; };
double DataStructure::median_time() { return t_median; };
double DataStructure::max_time() { return t_max; };

std::vector< std::vector<double> > DataStructure::data_2d(){
    std::vector< std::vector<double> > data_2d = std::vector< std::vector<double> >(N_datasets);
    for(int j = 0; j < N_datasets; j++){
        std::vector<double> dataset = std::vector<double>(M_datapoints);
        for(int i = 0; i < M_datapoints; i++){
            dataset[i] = _data[getVectorIndex(i, j)];
        }
        data_2d[j] = dataset;
    }
    return data_2d;
};

std::vector< std::vector<double> > DataStructure::err_2d(){
    if(err.empty()){
        return std::vector< std::vector<double> >(0);
    }
    std::vector< std::vector<double> > err_2d = std::vector< std::vector<double> >(N_datasets);
    for(int j = 0; j < N_datasets; j++){
        std::vector<double> dataset = std::vector<double>(M_datapoints);
        for(int i = 0; i < M_datapoints; i++){
            dataset[i] = err[getVectorIndex(i, j)];
        }
        err_2d[j] = dataset;
    }
    return err_2d;
};

std::vector< std::vector<double> > DataStructure::normalised_data_2d(){
    std::vector< std::vector<double> > data_2d = std::vector< std::vector<double> >(N_datasets);
    for(int j = 0; j < N_datasets; j++){
        std::vector<double> dataset = std::vector<double>(M_datapoints);
        for(int i = 0; i < M_datapoints; i++){
            dataset[i] = norm_data[getVectorIndex(i, j)];
        }
        data_2d[j] = dataset;
    }
    return data_2d;
};

