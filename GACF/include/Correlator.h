//
// Created by Josh Briegal on 23/11/2017.
//

#ifndef C_CORRELATOR_H
#define C_CORRELATOR_H

#include <cmath>
#include <vector>
#include <numeric>
#include <string>

#include "DataStructure.h"

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

struct CorrelationData{
    std::vector< std::vector<double> > _correlations; std::vector<double> _timeseries;
};

class Correlator;

//typedef double (Correlator::*MemberPointerType)(double, double);

class Correlator{

    DataStructure* ds;
    CorrelationData correlation_data;
    std::vector<double> N;

    double max_lag;
    double lag_resolution;
    double alpha; // characteristic length scale of weight functions
    double data_length;
    double num_data;

public:
    explicit Correlator(DataStructure*);
    Correlator(DataStructure*, DataStructure*); //X-correlation, to be implemented

    void naturalSelectionFunctionIdx(CorrelationIterator*);
    void fastSelectionFunctionIdx(CorrelationIterator*);
    void deltaT(CorrelationIterator*);
    void findCorrelation(CorrelationIterator*);
    void addCorrelationData(CorrelationIterator*);
//    void standardCorrelation(double, MemberPointerType, double);

    void clearCorrelation() {correlation_data = *new CorrelationData; };

    double fractionWeightFunction(double);
    double gaussianWeightFunction(double);

    void getFractionWeights(CorrelationIterator*);
    void getGaussianWeights(CorrelationIterator*);

    void setMaxLag(double);
    double getMaxLag();

    void setLagResolution(double);
    double getLagResolution();

    void setAlpha(double);
    double getAlpha();

    double getDataLength();
    double getNumData();

    std::vector<double>* rnormalised_timeseries();
    std::vector< std::vector<double> >* rvalues();
    std::vector<double>* rlag_timeseries();
    std::vector< std::vector<double> >* rcorrelations();

    std::vector<double> normalised_timeseries();
    std::vector< std::vector<double> > values();
    std::vector<double> lag_timeseries();
    std::vector< std::vector<double> > correlations();


};



#endif //C_CORRELATOR_H
