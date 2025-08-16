#include "sacf/DataStructure.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(datastructure, ds) {
    py::class_<DataStructure>(ds, "DataStructure")
        .def(py::init<const std::string &>())
        .def(py::init<std::vector<double>*, std::vector<double>* >())
        .def(py::init<std::vector<double>*, std::vector<double>*, std::vector<double>* >())
        .def(py::init<std::vector<double>*, std::vector< std::vector<double> >* >())
        .def(py::init<std::vector<double>*, std::vector< std::vector<double> >*, std::vector< std::vector<double> >* >())
        .def("data", &DataStructure::data_2d)
        .def("errors", &DataStructure::err_2d)
        .def("timeseries", &DataStructure::timeseries)
        .def("normalised_timeseries", &DataStructure::normalised_timeseries)
        .def("normalised_data", &DataStructure::normalised_data_2d)
        .def_property_readonly("mean_data", &DataStructure::mean_data)
        .def_property_readonly("median_time", &DataStructure::median_time)
        .def_property_readonly("max_time", &DataStructure::max_time)
        .def_property_readonly("N_datasets", [](const DataStructure &ds){ return ds.N_datasets; })
        .def_property_readonly("M_datapoints", [](const DataStructure &ds){ return ds.M_datapoints; })
        ;
    py::register_exception<EmptyDataStructureException>(ds, "EmptyDataStructureException");
    py::register_exception<BadDataFileReadException>(ds, "BadDataFileReadException");
    py::register_exception<BadDataInputException>(ds, "BadDataInputException");
}
