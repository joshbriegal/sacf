#include "DataStructure.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(datastructure, ds) {
    py::class_<DataStructure>(ds, "DataStructure")
        .def(py::init<const std::string &>())
        .def(py::init<std::vector<double>*, std::vector<double>* >())
        .def(py::init<std::vector<double>*, std::vector<double>*, std::vector<double>* >())
        .def("values", &DataStructure::values)
        .def("errors", &DataStructure::errors)
        .def("timeseries", &DataStructure::timeseries)
        .def("normalised_timeseries", &DataStructure::normalised_timeseries)
        .def("normalised_values", &DataStructure::normalised_values)
        .def("rnormalised_timeseries", &DataStructure::rnormalised_timeseries)
        .def("rnormalised_values", &DataStructure::rnormalised_values)
        .def_property_readonly("mean_X", &DataStructure::mean_X)
        .def_property_readonly("median_time", &DataStructure::median_time)
        .def_property_readonly("max_time", &DataStructure::max_time)
        ;
    py::register_exception<EmptyDataStructureException>(ds, "EmptyDataStructureException");
    py::register_exception<BadDataFileReadException>(ds, "BadDataFileReadException");
}
