//
// Created by Josh Briegal on 23/11/2017.
//

#ifndef C_DATASTRUCTURE_H
#define C_DATASTRUCTURE_H

#include <string>
#include <vector>

class EmptyDataStructureException: public std::exception{
    public: virtual const char* what() const throw();
};

class BadDataFileReadException: public std::exception{
    public: virtual const char* what() const throw();
};

class DataStructure{
    /*
     * Data and data-manipulation functions stored here. Can be initialised with or without data errors and from a file
     * containing this data in the format t, data, (data_error).
     */
    std::vector<double> t; std::vector< std::vector<double> > _data; std::vector< std::vector<double> > err;
    std::vector<double> data_mean; double t_median; double t_max; double t_length;
    std::vector<double> norm_t; std::vector< std::vector<double> > norm_data;

    void setDataMean();
    void settMedian();
    void settMax();
    void calcNormt();
    void calcNormData();

public:
    explicit DataStructure(const std::string &filename);
    DataStructure(std::vector<double>*, std::vector<double>*);
    DataStructure(std::vector<double>*, std::vector<double>*, std::vector<double>*);
    DataStructure(std::vector<double>*, std::vector< std::vector<double> >*);
    DataStructure(std::vector<double>*, std::vector< std::vector<double> >*, std::vector< std::vector<double> >*);

    // returnable values from the dataset
    std::vector< std::vector<double> >* rdata();
    std::vector< std::vector<double> >* rerrors();
    std::vector<double>* rtimeseries();
    std::vector<double>* rnormalised_timeseries();
    std::vector< std::vector<double> >* rnormalised_data();

    std::vector< std::vector<double> > data();
    std::vector< std::vector<double> > errors();
    std::vector<double> timeseries();
    std::vector<double> normalised_timeseries();
    std::vector< std::vector<double> > normalised_data();

    std::vector<double> mean_data();
    double median_time() ;
    double max_time();
};

template<typename T>
std::vector< std::vector<T> > convert_to_2d_vec(std::vector<T> vec_in){
    return std::vector< std::vector<T> >(1, vec_in);
};

#endif //C_DATASTRUCTURE_H
