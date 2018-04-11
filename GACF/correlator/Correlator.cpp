//
// Created by Josh Briegal on 23/11/2017.
//

#include "Correlator.h"
#include <cmath>

CorrelationIterator::CorrelationIterator(double k_in, double vec_size){
    k = k_in;
    correlation = std::vector<double>(vec_size);
    for(int i = 0; i < vec_size; i++){
        correlation[i] = 0;
    }
};

Correlator::Correlator(DataStructure* ds_in){
    ds = ds_in;
    num_data = ds->rdata()->size();
    data_length = ds->rnormalised_timeseries()->size();

    N = std::vector<double>(data_length);
    int i = 0;
    for(auto const& idata: ds->data()){
        for(auto const& X: idata){
            if(!std::isnan(X)){
                N[i] += (X - ds->mean_data()[i]) * (X - ds->mean_data()[i]);
            }
        }
        this->correlation_data._correlations.push_back(std::vector<double>(0));
        i++;
    }

    max_lag = ds->rnormalised_timeseries()->back();
    lag_resolution = max_lag / data_length;
    alpha = ds->median_time();
};

Correlator::Correlator(DataStructure *, DataStructure *) {
    /*
     * TODO
     */
}

std::vector<double>* Correlator::rnormalised_timeseries(){ return ds->rnormalised_timeseries(); }
std::vector< std::vector<double> >* Correlator::rvalues(){ return ds->rdata(); }
std::vector<double>* Correlator::rlag_timeseries() { return &correlation_data._timeseries; };
std::vector< std::vector<double> >* Correlator::rcorrelations() { return &correlation_data._correlations; };

std::vector<double> Correlator::normalised_timeseries(){ return ds->normalised_timeseries(); }
std::vector< std::vector<double> > Correlator::values(){ return ds->data(); }
std::vector<double> Correlator::lag_timeseries() { return correlation_data._timeseries; };
std::vector< std::vector<double> > Correlator::correlations() { return correlation_data._correlations; };

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
//            std::cout << "Index for shift = " << cor_it->shifted_timeseries[i] << ": " << idx;
            if(idx < 0){ idx = 0L; } //to prevent negative }
//            if(idx < cor_it->shifted_timeseries.size()-1) { // i.e. we haven't found the last value
//                if (abs(cor_it->shifted_timeseries[i] - ds->rnormalised_timeseries()->at(idx + 1))
//                    < abs(cor_it->shifted_timeseries[i] - ds->rnormalised_timeseries()->at(idx))) { idx = idx + 1; }
//            }
            if(idx != 0){ //i.e. not the first index
                if (abs(cor_it->shifted_timeseries[i] - ds->rnormalised_timeseries()->at(idx - 1))
                    < abs(cor_it->shifted_timeseries[i] - ds->rnormalised_timeseries()->at(idx))) { idx = idx - 1; }
            }
//            std::cout << " (" << idx << " adjusted)" << std::endl;
            cor_it->selection_indices.push_back(idx);
        }
    }
}

void Correlator::fastSelectionFunctionIdx(CorrelationIterator* cor_it){
    /*
     * returns a vector of indices corresponding to the closest time points for a shifted timeseries (shifted by k)
     * by finding the closest point for the first data point and filling in the rest by index.
     */
    for(auto const& time_point: *ds->rnormalised_timeseries()){
        if((time_point + cor_it->k) > ds->rnormalised_timeseries()->back()){
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
}

void Correlator::findCorrelation(CorrelationIterator* cor_it){
    int i = 0; int j = 0;
//    for(auto const& weight: cor_it->weights){
//        std::cout << "Calculating for point, weight " << i << " " << weight << std::endl;
//        j = 0;
//        for(auto const& idata: *ds->rnormalised_data()){
//            std::cout << "Calculating for series " << j << std::endl;
//            cor_it->correlation[j] += weight * idata[i] * idata[cor_it->selection_indices[i]];
//            j++;
//        }
//        i++;
//    }
//    std::cout << std::endl;


//
    for(auto const& idata: ds->normalised_data()){
        i = 0;
//        std::cout << "Calculating for series " << j << std::endl;
        for(auto const& weight: cor_it->weights){
//              std::cout << "Calculating for point, weight " << i << " " << weight << std::endl;
            cor_it->correlation[j] += weight * (idata[i]) *
                                   (idata[cor_it->selection_indices[i]]);
            i++;
        }
        cor_it->correlation[j] *= (1 / N[j]);
        j++;
    }
//      std::cout << std::endl;
}


double Correlator::fractionWeightFunction(double delta_t){
    return 1 / (1 + (delta_t / this->alpha));
}


double Correlator::gaussianWeightFunction(double delta_t){
    return exp( -((delta_t) * (delta_t)) / (2 * this->alpha * this->alpha));
}

void Correlator::deltaT(CorrelationIterator* cor_it){
    // returns a vector corresponding to the 'delta t' for each time point t_i + k subject to a selection index vector
    int t_i = 0;
    for(auto const& value: cor_it->selection_indices){
        cor_it->delta_t.push_back(abs(ds->rnormalised_timeseries()->at(value) - cor_it->shifted_timeseries[t_i]));
        t_i++;
    }
}

void Correlator::getFractionWeights(CorrelationIterator* cor_it){
    int i = 0;
    for(auto const& value: cor_it->delta_t){
        cor_it->weights.push_back(this->fractionWeightFunction(value));
        i++;
    }
}

void Correlator::getGaussianWeights(CorrelationIterator* cor_it){
    int i = 0;
    for(auto const& value: cor_it->delta_t){
        cor_it->weights.push_back(this->gaussianWeightFunction(value));
        i++;
    }
}

void Correlator::setMaxLag(double max_lag_in){
    this->max_lag = max_lag_in;
}
double Correlator::getMaxLag(){
    return this->max_lag;
}

void Correlator::setLagResolution(double lag_res){
    this->lag_resolution = lag_res;
}
double Correlator::getLagResolution(){
    return this->lag_resolution;
}

void Correlator::setAlpha(double alpha_in){
    this->alpha = alpha_in;
}
double Correlator::getAlpha(){
    return this->alpha;
}

double Correlator::getDataLength(){
    return this->data_length;
}

double Correlator::getNumData(){
    return this->num_data;
}

void Correlator::addCorrelationData(CorrelationIterator* col_it){
    this->correlation_data._timeseries.push_back(col_it->k);
    int i = 0;
    for(auto const& data_series_point: col_it->correlation){
        this->correlation_data._correlations[i].push_back(data_series_point);
        i++;
    }
//    this->correlation_data._correlations.push_back(col_it->correlation);
}

//void Correlator::standardCorrelation(double k, MemberPointerType weight_function, double alpha){
//    // define weight function & alpha (scale length of weight function)
//
//    auto const& col_it = new CorrelationIterator(k);
//    this->naturalSelectionFunctionIdx(col_it);
////    this->fastSelectionFunctionIdx(col_it);
//    this->deltaT(col_it);
//    this->getWeights(col_it, weight_function, alpha);
//    this->findCorrelation(col_it);
//    this->correlation_data.t.push_back(k);
//    this->correlation_data.X.push_back(col_it->correlation);
//    delete col_it;
//}


