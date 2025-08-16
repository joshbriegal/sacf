#include "sacf/Correlator.h"
#include "sacf/DataStructure.h"
#include "../datastructure/DataStructure.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(correlator, c) {

    py::class_<CorrelationIterator>(c, "CorrelationIterator")
        .def(py::init<double, double>())
        .def_readonly("correlation", &CorrelationIterator::correlation)
        .def_readwrite("k", &CorrelationIterator::k)
        .def_readonly("shifted_timeseries", &CorrelationIterator::shifted_timeseries)
        .def_readonly("selection_indices", &CorrelationIterator::selection_indices)
        .def_readonly("delta_t", &CorrelationIterator::delta_t)
        .def_readonly("weights", &CorrelationIterator::weights)
        ;

    py::class_<Correlator>(c, "Correlator")
        .def(py::init<DataStructure*>(), py::keep_alive<1, 2>())
        .def("naturalSelectionFunctionIdx", &Correlator::naturalSelectionFunctionIdx)
        .def("fastSelectionFunctionIdx", &Correlator::fastSelectionFunctionIdx)
        .def("deltaT", &Correlator::deltaT)
        .def("findCorrelation", &Correlator::findCorrelation)
        .def("clearCorrelation", &Correlator::clearCorrelation)
        .def("fractionWeightFunction", &Correlator::fractionWeightFunction)
        .def("fractionSquaredWeightFunction", &Correlator::fractionSquaredWeightFunction)
        .def("gaussianWeightFunction", &Correlator::gaussianWeightFunction)
        .def("getFractionWeights", &Correlator::getFractionWeights)
        .def("getFractionSquaredWeights", &Correlator::getFractionSquaredWeights)
        .def("getGaussianWeights", &Correlator::getGaussianWeights)
        .def("normalised_timeseries", &Correlator::normalised_timeseries)
        .def("values", &Correlator::values)
        .def("lag_timeseries", &Correlator::lag_timeseries)
        .def("correlations", &Correlator::correlations)
        .def("addCorrelationData", &Correlator::addCorrelationData)
        .def("cleanCorrelationData", &Correlator::cleanCorrelationData)
        .def("calculateStandardCorrelation", &Correlator::calculateStandardCorrelation)
        .def_property("max_lag", &Correlator::getMaxLag, &Correlator::setMaxLag)
        .def_property("min_lag", &Correlator::getMinLag, &Correlator::setMinLag)
        .def_property("lag_resolution", &Correlator::getLagResolution, &Correlator::setLagResolution)
        .def_property("alpha", &Correlator::getAlpha, &Correlator::setAlpha)
        .def_property_readonly("M_datapoints", &Correlator::getMDatapoints)
        .def_property_readonly("N_datasets", &Correlator::getNDatasets)
        ;

}