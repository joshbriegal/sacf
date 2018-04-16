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

class BadDataInputException: public std::exception{
    public: virtual const char* what() const throw();
};

class DataStructure{
    /*
     * Data and data-manipulation functions stored here. Can be initialised with or without data errors and from a file
     * containing this data in the format t, data, (data_error).
     */
    std::vector<double> t;
    std::vector<double> _data;
    std::vector<double> err;
    std::vector<double> data_mean;
    double t_median;
    double t_max;
    double t_length;
    std::vector<double> norm_t;
    std::vector<double> norm_data;

    void setDataMean();
    void settMedian();
    void settMax();
    void calcNormt();
    void calcNormData();
    void setUp();

public:
    explicit DataStructure(const std::string &filename);
    explicit DataStructure(std::vector<double>*, std::vector<double>*);
    explicit DataStructure(std::vector<double>*, std::vector<double>*, std::vector<double>*);
    explicit DataStructure(std::vector<double>*, std::vector< std::vector<double> >*);
    explicit DataStructure(std::vector<double>*, std::vector< std::vector<double> >*,
                           std::vector< std::vector<double> >*);

//    ~DataStructure();

    // indexing function
    long getVectorIndex(int i, int j);
    long N_datasets; // number of datasets, used for indexing [rows]
    long M_datapoints; // number of datapoints in each set, used for indexing [columns]

    // returnable values from the dataset
    std::vector<double>* rdata();
    std::vector<double>* rerrors();
    std::vector<double>* rtimeseries();
    std::vector<double>* rnormalised_timeseries();
    std::vector<double>* rnormalised_data();

    std::vector<double> data();
    std::vector<double> errors();
    std::vector<double> timeseries();
    std::vector<double> normalised_timeseries();
    std::vector<double> normalised_data();

    std::vector< std::vector<double> > data_2d();
    std::vector< std::vector<double> > err_2d();
    std::vector< std::vector<double> > normalised_data_2d();

    std::vector<double> mean_data();
    double median_time() ;
    double max_time();
};


//template<typename T>
//std::vector< std::vector<T> > convert_to_2d_vec(std::vector<T>* vec_in){
//    std::vector< std::vector<T> > temp = std::vector< std::vector<T> >(1, std::vector<T>(vec_in->size()));
//    temp.at(0) = *vec_in;
//    return temp;
//};

#endif //C_DATASTRUCTURE_H
