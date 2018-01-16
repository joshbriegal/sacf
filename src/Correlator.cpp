//
// Created by Josh Briegal on 23/11/2017.
//

#include "Correlator.h"


CorrelationIterator::CorrelationIterator(double k_in){
    k = k_in;
    correlation = 0;
};

Correlator::Correlator(DataStructure* data_in){
    data = data_in;
    N = 0;
    for(auto const& X: *data->values()){
        N += (X - data->mean_X()) * (X - data->mean_X());
    }
};

Correlator::Correlator(DataStructure *, DataStructure *) {
    /*
     * TODO
     */
}

std::vector<double>* Correlator::normalised_timeseries(){ return data->normalised_timeseries(); }
std::vector<double>* Correlator::values(){ return data->values(); }
std::vector<double>* Correlator::lag_timeseries() { return &correlation_data.t; };
std::vector<double>* Correlator::correlations() { return &correlation_data.X; };

void Correlator::naturalSelectionFunctionIdx(CorrelationIterator* cor_it){
    /*
     * returns a vector of indices corresponding to the closest time points for a shifted timeseries (shifted by k)
     * by finding the closest point for every data point.
     */
    for(auto const& time_point: *data->normalised_timeseries()){
        if((time_point + cor_it->k) > data->normalised_timeseries()->back()){
            break;
        }
        else { cor_it->shifted_timeseries.push_back(time_point + cor_it->k); }
    }

    if(!cor_it->shifted_timeseries.empty()) {
        for(int i = 0; i < cor_it->shifted_timeseries.size(); i++) {
            long idx = std::max(distance(data->normalised_timeseries()->begin(),
                                    lower_bound(data->normalised_timeseries()->begin(),
                                                data->normalised_timeseries()->end(),
                                                (double) cor_it->shifted_timeseries[i])) - 1, 0L); // to prevent negative
            if(idx < cor_it->shifted_timeseries.size()-1) { // i.e. we haven't found the last value!
                if (abs(cor_it->shifted_timeseries[i] - data->normalised_timeseries()->at(idx + 1))
                    < abs(cor_it->shifted_timeseries[i] - data->normalised_timeseries()->at(idx))) { idx = idx + 1; }
            }
            cor_it->selection_indices.push_back(idx);
        }
    }
}

void Correlator::fastSelectionFunctionIdx(CorrelationIterator* cor_it){
    /*
     * returns a vector of indices corresponding to the closest time points for a shifted timeseries (shifted by k)
     * by finding the closest point for the first data point and filling in the rest by index.
     */
    for(auto const& time_point: *data->normalised_timeseries()){
        if((time_point + cor_it->k) > data->normalised_timeseries()->back()){
            break;
        }
        else { cor_it->shifted_timeseries.push_back(time_point + cor_it->k); }
    }
    if(!cor_it->shifted_timeseries.empty()) {

        long idx = std::max(distance(data->normalised_timeseries()->begin(),
                                lower_bound(data->normalised_timeseries()->begin(), data->normalised_timeseries()->end(),
                                            (double) cor_it->shifted_timeseries[0])) - 1, 0L); // avoid negative indices
        long max_idx = cor_it->shifted_timeseries.size() - 1; // as we have ignored values shifted beyond time series already

        if (abs(cor_it->shifted_timeseries[0] - data->normalised_timeseries()->at(idx + 1))
            < abs(cor_it->shifted_timeseries[0] - data->normalised_timeseries()->at(idx))) { idx = idx + 1; }
        cor_it->selection_indices.resize(cor_it->shifted_timeseries.size());
        for (int i = 0; i <= max_idx - idx; i++) {
            cor_it->selection_indices[i] = idx + i;
        }
    }
}

void Correlator::findCorrelation(CorrelationIterator* cor_it){
    int i = 0;
    for(auto const& weight: cor_it->weights){
        cor_it->correlation += weight * (data->normalised_values()->at(i)) *
                               (data->normalised_values()->at(cor_it->selection_indices[i]));
        i++;
    }
    cor_it->correlation *= (1 / N);
}


double Correlator::fractionWeightFunction(double delta_t, double alpha){
    return 1 / (1 + (delta_t / alpha));
}


double Correlator::gaussianWeightFunction(double delta_t, double sigma){
    return exp( -((delta_t) * (delta_t)) / (2 * sigma * sigma));
}

void Correlator::deltaT(CorrelationIterator* cor_it){
    // returns a vector corresponding to the 'delta t' for each time point t_i + k subject to a selection index vector
    int t_i = 0;
    for(auto const& value: cor_it->selection_indices){
        cor_it->delta_t.push_back(abs(data->normalised_timeseries()->at(value) - cor_it->shifted_timeseries[t_i]));
        t_i++;
    }
}

void Correlator::getWeights(CorrelationIterator* cor_it, MemberPointerType weight_function,
                            double alpha = 0.){
    int i = 0;
    for(auto const& value: cor_it->delta_t){
        cor_it->weights.push_back((this->*weight_function)(value, alpha));
        i++;
    }
}

void Correlator::standardCorrelation(double k, MemberPointerType weight_function, double alpha){
    // define weight function & alpha (scale length of weight function)

    auto const& col_it = new CorrelationIterator(k);
    this->naturalSelectionFunctionIdx(col_it);
//    this->fastSelectionFunctionIdx(col_it);
    this->deltaT(col_it);
    this->getWeights(col_it, weight_function, alpha);
    this->findCorrelation(col_it);
    this->correlation_data.t.push_back(k);
    this->correlation_data.X.push_back(col_it->correlation);
    delete col_it;
}


