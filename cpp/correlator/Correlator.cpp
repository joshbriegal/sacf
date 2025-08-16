//
// Created by Josh Briegal on 23/11/2017.
//

#include "sacf/Correlator.h"
#include <cmath>
#include <iostream>
#include <cfloat>

double findMinDiff(std::vector<double>* arr){
    // Initialize difference as infinite
   double diff = DBL_MAX;
   int n = arr->size();

   // Find the min diff by comparing adjacent pairs in sorted array
   for (int i=0; i<n-1; i++){
      double tdiff = arr->at(i+1) - arr->at(i);
      if (tdiff < diff){
          diff = tdiff;
      }
   }
   return diff;
}

CorrelationIterator::CorrelationIterator(double k_in, double vec_size){
    k = k_in;
    correlation = std::vector<double>(vec_size);
    for(int i = 0; i < vec_size; i++){
        correlation[i] = 0;
    }
};

Correlator::Correlator(DataStructure* ds_in){
    ds = ds_in;
    N_datasets = ds->N_datasets;
    M_datapoints = ds->M_datapoints;

    N = std::vector<double>(N_datasets);
    int i = 0;
    for(int j = 0; j < N_datasets; j++){
        for(int i = 0; i < M_datapoints; i++){
            float X = ds->rdata()->at(ds->getVectorIndex(i, j));
            if(!std::isnan(X)){
                N[j] += (X - ds->mean_data()[j]) * (X - ds->mean_data()[j]);
            }
        }
    }
    max_lag = ds->rnormalised_timeseries()->back();
    min_lag = -max_lag;
    lag_resolution = findMinDiff(ds->rnormalised_timeseries());
    alpha = ds->median_time();
    num_lag_steps = (int) std::floor((max_lag - min_lag) / lag_resolution) + 3; // 3 additional elements at min, 0, max
    correlation_data._correlations = std::vector<double>(N_datasets * num_lag_steps);
    correlation_data._timeseries = std::vector<double>(num_lag_steps);
};

Correlator::Correlator(DataStructure *, DataStructure *) {
    /*
     * TODO
     */
}

std::vector<double> Correlator::normalised_timeseries(){ return ds->normalised_timeseries(); }
std::vector< std::vector<double> > Correlator::values(){ return ds->data_2d(); }
std::vector<double> Correlator::lag_timeseries() { return correlation_data._timeseries; };

std::vector< std::vector<double> > Correlator::correlations() {
    std::vector< std::vector<double> > correlations = std::vector< std::vector<double> >(N_datasets);
    for(int j = 0; j < N_datasets; j++){
        std::vector<double> dataset = std::vector<double>(num_lag_steps);
        for(int i = 0; i < num_lag_steps; i++){
            dataset[i] = correlation_data._correlations[i + j*num_lag_steps];
        }
        correlations[j] = dataset;
    }
    return correlations;
};

void Correlator::naturalSelectionFunctionIdx(CorrelationIterator* cor_it){
    /*
     * returns a vector of indices corresponding to the closest time points for a shifted timeseries (shifted by k)
     * by finding the closest point for every data point.
     */
    for(auto const& time_point: *ds->rnormalised_timeseries()){
        if((time_point + cor_it->k) > ds->rnormalised_timeseries()->back()){
            break;
        }
        else { cor_it->shifted_timeseries.push_back(time_point + cor_it->k); }
    }

    if(!cor_it->shifted_timeseries.empty()) {
        for(int i = 0; i < cor_it->shifted_timeseries.size(); i++) {
            long idx = distance(ds->rnormalised_timeseries()->begin(),
                                    lower_bound(ds->rnormalised_timeseries()->begin(),
                                                ds->rnormalised_timeseries()->end(),
                                                (double) cor_it->shifted_timeseries[i]));
            if(idx < 0){ idx = 0L; } //to prevent negative 

            if(idx != 0){ //i.e. not the first index
                if (abs(cor_it->shifted_timeseries[i] - ds->rnormalised_timeseries()->at(idx - 1))
                    < abs(cor_it->shifted_timeseries[i] - ds->rnormalised_timeseries()->at(idx))) { idx = idx - 1; }
            }
            cor_it->selection_indices.push_back(idx);
        }
    }
    cor_it->shifted_timeseries.clear();
}

void Correlator::fastSelectionFunctionIdx(CorrelationIterator* cor_it){
    /*
     * returns a vector of indices corresponding to the closest time points for a shifted timeseries (shifted by k)
     * by finding the closest point for the first data point and filling in the rest by index.
     */
    for(auto const& time_point: *ds->rnormalised_timeseries()){
        if((time_point + cor_it->k) >= ds->rnormalised_timeseries()->back()){
            break;
        }
        else { cor_it->shifted_timeseries.push_back(time_point + cor_it->k); }
    }
    if(!cor_it->shifted_timeseries.empty()) {

        long idx = std::max(distance(ds->rnormalised_timeseries()->begin(),
                                lower_bound(ds->rnormalised_timeseries()->begin(), ds->rnormalised_timeseries()->end(),
                                            (double) cor_it->shifted_timeseries[0])), 0L); // avoid negative indices
        long max_idx = cor_it->shifted_timeseries.size() - 1; // as we have ignored ds shifted beyond time series already

        if (abs(cor_it->shifted_timeseries[0] - ds->rnormalised_timeseries()->at(idx + 1))
            < abs(cor_it->shifted_timeseries[0] - ds->rnormalised_timeseries()->at(idx))) { idx = idx + 1; }
        cor_it->selection_indices.resize(cor_it->shifted_timeseries.size());
        for (int i = 0; i <= max_idx - idx; i++) {
            cor_it->selection_indices[i] = idx + i;
        }
    }
    cor_it->shifted_timeseries.clear();
}

void Correlator::findCorrelation(CorrelationIterator* cor_it){
    for(int j = 0; j < N_datasets; j++){
        int i = 0;
        for(auto const& weight: cor_it->weights){
            cor_it->correlation[j] += weight * (ds->rnormalised_data()->at(ds->getVectorIndex(i, j))) *
                                      (ds->rnormalised_data()->at(ds->getVectorIndex(cor_it->selection_indices[i], j)));
            i++;
        }
        cor_it->correlation[j] *= (1 / N[j]);
    }
}


double Correlator::fractionWeightFunction(double delta_t){
    return 1 / (1 + (delta_t / alpha));
}

double Correlator::fractionSquaredWeightFunction(double delta_t){
    return 1 / (1 + (delta_t / alpha) * (delta_t / alpha));
}

double Correlator::gaussianWeightFunction(double delta_t){
    return exp( -((delta_t) * (delta_t)) / (2 * alpha * alpha));
}

void Correlator::deltaT(CorrelationIterator* cor_it){
    // returns a vector corresponding to the 'delta t' for each time point t_i + k subject to a selection index vector
    int t_i = 0;
    for(auto const& value: cor_it->selection_indices){
        cor_it->delta_t.push_back(abs(ds->rnormalised_timeseries()->at(value) - cor_it->shifted_timeseries[t_i]));
        t_i++;
    }
    cor_it->shifted_timeseries.clear();
}

void Correlator::getFractionWeights(CorrelationIterator* cor_it){
    int i = 0;
    for(auto const& value: cor_it->delta_t){
        cor_it->weights.push_back(fractionWeightFunction(value));
        i++;
    }
    cor_it->delta_t.clear();
}

void Correlator::getFractionSquaredWeights(CorrelationIterator* cor_it){
    int i = 0;
    for(auto const& value: cor_it->delta_t){
        cor_it->weights.push_back(fractionSquaredWeightFunction(value));
        i++;
    }
    cor_it->delta_t.clear();
}

void Correlator::getGaussianWeights(CorrelationIterator* cor_it){
    int i = 0;
    for(auto const& value: cor_it->delta_t){
        cor_it->weights.push_back(gaussianWeightFunction(value));
        i++;
    }
    cor_it->delta_t.clear();
}

void Correlator::setMaxLag(double max_lag_in){
    max_lag = max_lag_in;
}
double Correlator::getMaxLag(){
    return max_lag;
}

void Correlator::setMinLag(double min_lag_in){
    min_lag = min_lag_in;
}
double Correlator::getMinLag(){
    return min_lag;
}

void Correlator::setLagResolution(double lag_res){
    lag_resolution = lag_res;
}
double Correlator::getLagResolution(){
    return lag_resolution;
}

void Correlator::setAlpha(double alpha_in){
    alpha = alpha_in;
}
double Correlator::getAlpha(){
    return alpha;
}

double Correlator::getMDatapoints(){
    return M_datapoints;
}

double Correlator::getNDatasets(){
    return N_datasets;
}

void Correlator::addCorrelationData(CorrelationIterator* col_it, int timestep_number){
    correlation_data._timeseries[timestep_number] = col_it->k;
    for(int j = 0; j < N_datasets; j++){
        correlation_data._correlations[timestep_number + num_lag_steps * j] = col_it->correlation[j];
    }
}

void Correlator::cleanCorrelationData(int i){
    // clean up extra elements at end of vector not used.
    correlation_data._timeseries.resize(i);
    std::vector<double> copy_correlations = std::vector<double>(N_datasets * i);
    for(int j = 0; j < N_datasets; j++){
        auto start = correlation_data._correlations.begin() + (j * num_lag_steps);
        std::copy(start, start + i, copy_correlations.begin() + (j * i));
    }
    correlation_data._correlations = copy_correlations;
    num_lag_steps = i;
}

void Correlator::calculateStandardCorrelation(){

    double k = min_lag;
    bool is_positive = false;
    int i = 0;
    while(k <= max_lag){
        if(k==0){
            is_positive = true;
        }
        if((k > 0) && (!is_positive)){
            // we have gone beyond 0, so calculate for correlation 0 and increment counter.
            _calculateStandardCorrelation(0, i);
            is_positive = true;
            i++;
        }
        _calculateStandardCorrelation(k, i);
        k += lag_resolution;
        i++;
    }
    cleanCorrelationData(i);
}

void Correlator::_calculateStandardCorrelation(double k, int i){
    /*
    Standard method of calculating correlations, using fractional weight function and natural selection function
    */
    auto const& col_it = new CorrelationIterator(k, N_datasets);
    naturalSelectionFunctionIdx(col_it);
    deltaT(col_it);
    getFractionWeights(col_it);
    findCorrelation(col_it);
    addCorrelationData(col_it, i);
    delete col_it;
}
