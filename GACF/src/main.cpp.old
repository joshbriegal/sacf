#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <ctime>
#include <random>

#include "DataStructure.h"
#include "Correlator.h"

#define _USE_MATH_DEFINES

using namespace std;

void data_output_to_file(string filename, vector<double>* timeseries, vector<double>* values){
    ofstream output_file(filename);
    output_file << "time" << "\t" << "value" << endl;
    for(int i = 0; i < timeseries->size() and i< values->size(); i++){
        output_file << timeseries->at(i) << "\t" << values->at(i) << endl;
    }

}

void createCorrelation(Correlator* col, double n_values, MemberPointerType weight_function, double alpha){
    double k = 0.;
    while(k < 30.){
          // col->normalised_timeseries()->back(){
        col->standardCorrelation(k, weight_function, alpha);
//        k += col->normalised_timeseries()->back() / n_values;
        k += 30. / n_values;
    }
}

void correlationsToFile(Correlator* col, double alpha, std::string custom_string = ""){
//    char buffer[80];
//    time_t rawtime;
//    const tm* current_time;
//    time(&rawtime);
//    current_time = localtime(&rawtime);
//    std::strftime(buffer, 80, "%F-%H%M%S",current_time);
//
//    std::string sample_file_name = "sampled" + (std::string)buffer + ".txt";
//
//    std::string correlation_file_name = "correlations_sampled" + (std::string)buffer + ".txt";

    std::string sample_file_name = "sampled_" + custom_string + "_" + std::to_string(alpha) + ".txt";
    std::string correlation_file_name = "correlations_sampled_" + custom_string + "_" + std::to_string(alpha) + ".txt";

    data_output_to_file(sample_file_name, col->normalised_timeseries(), col->values());
    data_output_to_file(correlation_file_name, col->lag_timeseries(), col->correlations());
}

int main() {
    /*
     * From file
     */
    string filename = "../../../../files/NG0522-2518_025520_LC_tbin=10min.dat";
    DataStructure test_import = DataStructure(filename);

    /*
     * Generated data
     */
//    vector<double> t_values;
//    double t_length = test_import.normalised_timeseries()->back();
    int n_values = 10000;
//    double gap_length_days = 1.5;
//    double sampling_length_minutes = 10.;
//    int jump_count = 0;
//    for(int i = 0; i<=n_values; i++){
//        if(i%72 == 0) {
//            jump_count++;
//        }
//        t_values.push_back(i * (sampling_length_minutes / (60. * 24.)));// + jump_count * gap_length_days);
//        t_values.push_back(max((double)i*t_length/n_values + (((double)(rand() % 1) - 0.5) * 100/n_values),0.)); //completely random sampling
//    }
    vector<double> x_values;// = *test_import.values();
    vector<double> t_values = *test_import.timeseries();
    transform(t_values.begin(), t_values.end(), back_inserter(x_values), [](double t) -> double {
        return cos(t * (2.*M_PI / 26.8)); });
//    std::random_device generator;
//    std::normal_distribution<double> distribution (0.,1.0);
//    for(int i = 0; i < t_values.size(); i++){
//        x_values.push_back(distribution(generator));
//    }
//    transform(t_values.begin(), t_values.end(), back_inserter(x_values), [generator, distribution](double t)-> double { return distribution(generator); });

    //create data structure
    DataStructure test = DataStructure(&x_values, &t_values);

    //create correlator using above data
    Correlator test_col = Correlator(&test);


    //set up different values of alpha to test
    vector<double> alpha_values;
//    alpha_values.push_back(0.001);
//    alpha_values.push_back(0.01);
//    alpha_values.push_back(0.1);
//    alpha_values.push_back(1.0);
    alpha_values.push_back(10.0);
//    alpha_values.push_back(test.median_time());
//    alpha_values.push_back(.5);
//    alpha_values.push_back(.6);
//    alpha_values.push_back(.7);
//    alpha_values.push_back(.8);
//    alpha_values.push_back(.9);
//    alpha_values.push_back(1.01);
//    alpha_values.push_back(1.02);
//    alpha_values.push_back(1.03);
//    alpha_values.push_back(1.04);
//    alpha_values.push_back(1.05);



    for(auto const& alpha: alpha_values){

        test_col.clearCorrelation(); // clear previous correlation data from data

        MemberPointerType weight_function= &Correlator::fractionWeightFunction;

        createCorrelation(&test_col, n_values,  weight_function, alpha);
        correlationsToFile(&test_col, alpha, "fractionWeightFunction");
        cout << "alpha value " << alpha << " complete for fractionWeightFunction" << endl;
    }

    for(auto const& alpha: alpha_values){

        test_col.clearCorrelation(); // clear previous correlation data from data

        MemberPointerType weight_function= &Correlator::gaussianWeightFunction;

        createCorrelation(&test_col, n_values,  weight_function, alpha);
        correlationsToFile(&test_col, alpha, "gaussianWeightFunction");
        cout << "alpha value " << alpha << " complete for gaussianWeightFunction" << endl;
    }

//    double k = 0.;
//    clock_t start, end, diff;
//    start = clock();
//    while(k < test.normalised_timeseries()->back()){
//        test_col.standardCorrelation(k);
//        k += test.normalised_timeseries()->back() / n_values;
//    }
//    end = clock();
//    diff = ((double) (end - start)) / CLOCKS_PER_SEC;
//
//    cout << "natural selection function took " << diff << "s" << endl;
//    cout << "for " << t_values.size() << " values" << endl;

//    data_output_to_file("cos.txt", *test.normalised_timeseries(), *test.values());
//    data_output_to_file("correlations.txt", test_col.lag_timeseries(), test_col.correlations());


    /*
     * to time the index matching
     */
//    clock_t startN, endN, diffN;
//    clock_t startF, endF, diffF;
//    startN = clock();
//    vector<long> indices_natural = test_col.naturalSelectionFunctionIdx(0.6);
//    endN = clock();
//    startF = clock();
//    vector<long> indices_fast = test_col.fastSelectionFunctionIdx(0.6);
//    endF = clock();
//    diffN = ((double) (endN - startN)) * 1000 / CLOCKS_PER_SEC;
//    diffF = ((double) (endF - startF)) * 1000 / CLOCKS_PER_SEC;
//    cout << "natural selection function took " << diffN << "ms" << endl;
//    cout << "fast selection function took " << diffF << "ms" << endl;
//    cout << "for " << t_values.size() << " values" << endl;


    //data_output_to_file("sin.txt", t_values, x_values);

    return 0;
}
