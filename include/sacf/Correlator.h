//
// Created by Josh Briegal on 23/11/2017.
//

#ifndef C_CORRELATOR_H
#define C_CORRELATOR_H

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#ifndef __cplusplus
  #error C++ is required
#elif __cplusplus < 201402L
  #pragma message(TOSTRING(__cplusplus))
  #error C++14 is required
#endif


#include <cmath>
#include <vector>
#include <numeric>
#include <string>

#include "sacf/DataStructure.h"

double findMinDiff(std::vector<double>*);


struct CorrelationIterator{
    // struct containing all useful data per k-step iteration
    double k;
    std::vector<double> correlation;
    std::vector<double> shifted_timeseries;
    std::vector<long> selection_indices;
    std::vector<double> delta_t;
    std::vector<double> weights;

    explicit CorrelationIterator(double, double);
};


template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& v) {
    std::size_t total_size = 0;
    for (const auto& sub : v)
        total_size += sub.size(); // I wish there was a transform_accumulate
    std::vector<T> result;
    result.reserve(total_size);
    for (const auto& sub : v)
        result.insert(result.end(), sub.begin(), sub.end());
    return result;
}

struct CorrelationData{
    std::vector<double> _correlations; std::vector<double> _timeseries;
};

class Correlator;

//typedef double (Correlator::*MemberPointerType)(double, double);

class Correlator{

    DataStructure* ds;
    CorrelationData correlation_data;
    std::vector<double> N;

    double max_lag;
    double min_lag;
    double lag_resolution;
    double alpha; // characteristic length scale of weight functions
    double N_datasets;
    double M_datapoints;
    int num_lag_steps;

public:
    explicit Correlator(DataStructure*);
    Correlator(DataStructure*, DataStructure*); //X-correlation, to be implemented

    void naturalSelectionFunctionIdx(CorrelationIterator*);
    void fastSelectionFunctionIdx(CorrelationIterator*);
    void deltaT(CorrelationIterator*);
    void findCorrelation(CorrelationIterator*);
    void addCorrelationData(CorrelationIterator*, int);
    void cleanCorrelationData(int);
    void calculateStandardCorrelation();
    void _calculateStandardCorrelation(double, int);

    void clearCorrelation() {correlation_data = *new CorrelationData; };

    double fractionWeightFunction(double);
    double fractionSquaredWeightFunction(double);
    double gaussianWeightFunction(double);

    void getFractionWeights(CorrelationIterator*);
    void getFractionSquaredWeights(CorrelationIterator*);
    void getGaussianWeights(CorrelationIterator*);

    void setMaxLag(double);
    double getMaxLag();

    void setMinLag(double);
    double getMinLag();

    void setLagResolution(double);
    double getLagResolution();

    void setAlpha(double);
    double getAlpha();

    double getNDatasets();
    double getMDatapoints();

//    std::vector<double>* rnormalised_timeseries();
//    std::vector< std::vector<double> >* rvalues();
//    std::vector<double>* rlag_timeseries();
//    std::vector< std::vector<double> >* rcorrelations();

    std::vector<double> normalised_timeseries();
    std::vector< std::vector<double> > values();
    std::vector<double> lag_timeseries();
    std::vector< std::vector<double> > correlations();


};



#endif //C_CORRELATOR_H
