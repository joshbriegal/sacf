from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import sacf
import sacf.datastructure
import sacf.correlator
import numpy as np
import os

from sacf import SACF

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestDataStructure(unittest.TestCase):
    """
    Test class for C++ DataStructure object

    Test import cases (from lists / from file)
    Test exposed methods and properties work as expected (with NaN values included)
    """

    integer_list = [0, 1, 2, 3, 4]
    integer_list_mean = 2
    integer_list_median = 2
    short_integer_list = [0, 1, 2]
    integer_list2 = [10, 11, 12, 13, 14]
    integer_list2_mean = 12
    integer_list2_median = 12
    float_list = [0.0, 1.0, 2.0, 3.0, 4.0]
    float_list_mean = 2.0
    float_list_median = 2.0
    numpy_float_array = np.array(float_list)
    empty_list = []
    empty_numpy_array = np.array([])
    numpy_float_array_with_nan = np.append(numpy_float_array, np.nan)
    numpy_float_array_with_nan_median = 2.5

    # tests for 2 list constructor

    def test_import_integer_lists(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.integer_list, TestDataStructure.integer_list
        )

    def test_import_float_lists(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.float_list, TestDataStructure.float_list
        )

    def test_import_numpy_float_array(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.numpy_float_array, TestDataStructure.numpy_float_array
        )

    def test_import_empty_list(self):
        with self.assertRaises(sacf.datastructure.EmptyDataStructureException):
            ds = sacf.datastructure.DataStructure(
                TestDataStructure.empty_list, TestDataStructure.empty_list
            )

    def test_import_empty_numpy_array(self):
        with self.assertRaises(sacf.datastructure.EmptyDataStructureException):
            ds = sacf.datastructure.DataStructure(
                TestDataStructure.empty_numpy_array, TestDataStructure.empty_numpy_array
            )

    def test_import_numpy_float_array_with_nan(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.numpy_float_array_with_nan,
            TestDataStructure.numpy_float_array_with_nan,
        )

    def test_uneven_lists(self):
        with self.assertRaises(sacf.datastructure.BadDataInputException):
            ds = sacf.datastructure.DataStructure(
                TestDataStructure.short_integer_list, TestDataStructure.integer_list
            )

    def test_uneven_lists2(self):
        with self.assertRaises(sacf.datastructure.BadDataInputException):
            ds = sacf.datastructure.DataStructure(
                TestDataStructure.integer_list, TestDataStructure.short_integer_list
            )

    # Tests for 3 list constructor

    def test_import_integer_lists_with_err(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.integer_list,
            TestDataStructure.integer_list,
            TestDataStructure.integer_list,
        )

    def test_import_float_lists_with_err(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.float_list,
            TestDataStructure.float_list,
            TestDataStructure.float_list,
        )

    def test_import_numpy_float_array_with_err(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.numpy_float_array,
            TestDataStructure.numpy_float_array,
            TestDataStructure.numpy_float_array,
        )

    def test_import_empty_list_with_err(self):
        with self.assertRaises(sacf.datastructure.EmptyDataStructureException):
            ds = sacf.datastructure.DataStructure(
                TestDataStructure.empty_list,
                TestDataStructure.empty_list,
                TestDataStructure.empty_list,
            )

    def test_import_empty_numpy_array_with_err(self):
        with self.assertRaises(sacf.datastructure.EmptyDataStructureException):
            ds = sacf.datastructure.DataStructure(
                TestDataStructure.empty_numpy_array,
                TestDataStructure.empty_numpy_array,
                TestDataStructure.empty_numpy_array,
            )

    def test_import_numpy_float_array_with_nan_with_err(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.numpy_float_array_with_nan,
            TestDataStructure.numpy_float_array_with_nan,
            TestDataStructure.numpy_float_array_with_nan,
        )

    # Tests for file based constructor

    def test_import_csv_with_title(self):
        ds = sacf.datastructure.DataStructure(
            os.path.join(THIS_DIR, "csv_with_titles.csv")
        )

    def test_import_csv_with_title_commented(self):
        ds = sacf.datastructure.DataStructure(
            os.path.join(THIS_DIR, "csv_with_titles_commented.csv")
        )

    def test_import_csv(self):
        ds = sacf.datastructure.DataStructure(
            os.path.join(THIS_DIR, "csv_no_titles.csv")
        )

    def test_import_csv_with_err(self):
        ds = sacf.datastructure.DataStructure(
            os.path.join(THIS_DIR, "csv_no_titles_with_err.csv")
        )

    def test_import_csv_with_err_with_title(self):
        ds = sacf.datastructure.DataStructure(
            os.path.join(THIS_DIR, "csv_with_titles_with_err.csv")
        )

    def test_import_tab_delim_file_with_title(self):
        ds = sacf.datastructure.DataStructure(
            os.path.join(THIS_DIR, "tab_delim_with_titles.txt")
        )

    def test_import_tab_delim_file_with_title_commented(self):
        ds = sacf.datastructure.DataStructure(
            os.path.join(THIS_DIR, "tab_delim_with_titles_commented.txt")
        )

    def test_import_tab_delim_file(self):
        ds = sacf.datastructure.DataStructure(
            os.path.join(THIS_DIR, "tab_delim_no_titles.txt")
        )

    def test_import_tab_delim_file_with_err(self):
        ds = sacf.datastructure.DataStructure(
            os.path.join(THIS_DIR, "tab_delim_no_titles_with_err.txt")
        )

    def test_import_tab_delim_file_with_err_with_title(self):
        ds = sacf.datastructure.DataStructure(
            os.path.join(THIS_DIR, "tab_delim_with_titles_with_err.txt")
        )

    # method & property tests

    def test_data(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.integer_list, TestDataStructure.integer_list2
        )
        self.assertEqual(TestDataStructure.integer_list2, ds.data()[0])

    def test_data_2d(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.integer_list,
            [TestDataStructure.integer_list2, TestDataStructure.integer_list],
        )
        self.assertEqual(
            [TestDataStructure.integer_list2, TestDataStructure.integer_list], ds.data()
        )

    def test_timeseries(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.integer_list2, TestDataStructure.integer_list
        )
        self.assertEqual(TestDataStructure.integer_list2, ds.timeseries())

    def test_errors_empty(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.integer_list2, TestDataStructure.integer_list
        )
        self.assertEqual(TestDataStructure.empty_list, ds.errors())

    def test_errors(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.integer_list,
            TestDataStructure.integer_list,
            TestDataStructure.integer_list2,
        )
        self.assertEqual(TestDataStructure.integer_list2, ds.errors()[0])

    def test_normalised_data(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.integer_list, TestDataStructure.integer_list2
        )
        self.assertEqual(
            [
                x - TestDataStructure.integer_list2_mean
                for x in TestDataStructure.integer_list2
            ],
            ds.normalised_data()[0],
        )

    def test_normalised_timeseries(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.integer_list2, TestDataStructure.integer_list
        )
        self.assertEqual(TestDataStructure.integer_list, ds.normalised_timeseries())

    def test_mean_data(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.integer_list, TestDataStructure.integer_list2
        )
        self.assertEqual(TestDataStructure.integer_list2_mean, ds.mean_data[0])

    def test_mean_data_nan(self):
        ds = sacf.datastructure.DataStructure(
            np.append(TestDataStructure.integer_list, 4),
            TestDataStructure.numpy_float_array_with_nan,
        )
        self.assertEqual(TestDataStructure.float_list_mean, ds.mean_data[0])

    def test_median_time(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.float_list, TestDataStructure.integer_list
        )
        self.assertEqual(TestDataStructure.float_list_median, ds.median_time)

    def test_median_time_nan(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.numpy_float_array_with_nan,
            np.append(TestDataStructure.integer_list, 4),
        )
        self.assertEqual(
            TestDataStructure.numpy_float_array_with_nan_median, ds.median_time
        )

    def test_max_time(self):
        ds = sacf.datastructure.DataStructure(
            TestDataStructure.float_list, TestDataStructure.integer_list
        )
        self.assertEqual(TestDataStructure.float_list[-1], ds.max_time)


class TestCorrelator(unittest.TestCase):
    def setUp(self):
        self.timestamps1 = [0, 1, 2, 3, 4]
        self.timestamps1_median = 2
        self.data1 = [0, 1, 0, -1, 0]
        self.data2 = [0, 1, 2, 1, np.nan]
        self.ds1 = sacf.datastructure.DataStructure(self.timestamps1, self.data1)
        self.ds2 = sacf.datastructure.DataStructure(
            self.timestamps1, [self.data1, self.data2]
        )
        self.simple_correlations_lag_timeseries = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.simple_correlation_solution = [1.0, 0.0, -0.5, 0.0, 0.0]
        self.two_correlation_solution = [
            [1.0, 0.0, -0.5, 0.0, 0.0],
            [1.0, 0.0, -0.5, 0.0, 0.0],
        ]
        self.negative_correlations_lag_timeseries = [
            -4.0,
            -3.0,
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
        ]
        self.negative_correlations_solution = [
            0.0,
            0.0,
            -0.5,
            0.0,
            1.0,
            0.0,
            -0.5,
            0.0,
            0.0,
        ]
        self.add_zero_lag_timeseries = [-4.0, -2.5, -1.0, 0.0, 0.5, 2.0, 3.5]

    def test_setup_correlator(self):
        corr = sacf.correlator.Correlator(self.ds1)

    def test_setup_correlationiterator(self):
        cor_it = sacf.correlator.CorrelationIterator(1, 1)

    def test_default_max_lag(self):
        corr = sacf.correlator.Correlator(self.ds1)
        self.assertEqual(self.timestamps1[-1], corr.max_lag)

    def test_alter_max_lag(self):
        max_lag = 100
        corr = sacf.correlator.Correlator(self.ds1)
        corr.max_lag = max_lag
        self.assertEqual(max_lag, corr.max_lag)

    def test_alter_min_lag(self):
        min_lag = -100
        corr = sacf.correlator.Correlator(self.ds1)
        corr.min_lag = min_lag
        self.assertEqual(min_lag, corr.min_lag)

    def test_default_lag_resolution(self):
        corr = sacf.correlator.Correlator(self.ds1)
        self.assertEqual(1.0, corr.lag_resolution)

    def test_alter_lag_resolution(self):
        lag_res = 100
        corr = sacf.correlator.Correlator(self.ds1)
        corr.lag_resolution = 100
        self.assertEqual(lag_res, corr.lag_resolution)

    def test_default_alpha(self):
        corr = sacf.correlator.Correlator(self.ds1)
        self.assertEqual(self.timestamps1_median, corr.alpha)

    def test_alter_alpha(self):
        alpha = 100
        corr = sacf.correlator.Correlator(self.ds1)
        corr.alpha = 100
        self.assertEqual(alpha, corr.alpha)


class TestSACF(unittest.TestCase):
    def setUp(self):
        self.timestamps1 = [0, 1, 2, 3, 4]
        self.timestamps1_median = 2
        self.data1 = [0, 1, 0, -1, 0]
        self.data2 = [0, 1, 2, 1, np.nan]
        self.ds1 = sacf.datastructure.DataStructure(self.timestamps1, self.data1)
        self.ds2 = sacf.datastructure.DataStructure(
            self.timestamps1, [self.data1, self.data2]
        )
        self.simple_correlations_lag_timeseries = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.simple_correlation_solution = [1.0, 0.0, -0.5, 0.0, 0.0]
        self.two_correlation_solution = [
            [1.0, 0.0, -0.5, 0.0, 0.0],
            [1.0, 0.0, -0.5, 0.0, 0.0],
        ]
        self.negative_correlations_lag_timeseries = [
            -4.0,
            -3.0,
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
        ]
        self.negative_correlations_solution = [
            0.0,
            0.0,
            -0.5,
            0.0,
            1.0,
            0.0,
            -0.5,
            0.0,
            0.0,
        ]
        self.add_zero_lag_timeseries = [-4.0, -2.5, -1.0, 0.0, 0.5, 2.0, 3.5]

    def test_SACF_init(self):
        sacf_instance = SACF(self.timestamps1, self.data1)

        self.assertIsInstance(sacf_instance.data, sacf.datastructure.DataStructure)

    def test_SACF_correlation(self):
        lag_timeseries, correlations = SACF(
            self.timestamps1, self.data1
        ).autocorrelation()

        self.assertEqual(self.negative_correlations_lag_timeseries, lag_timeseries)
        self.assertEqual(self.negative_correlations_solution, correlations)

    def test_SACF_correlation_return_corr(self):
        lag_timeseries, correlations, corr = SACF(
            self.timestamps1, self.data1
        ).autocorrelation(return_correlator=True)

        self.assertEqual(self.negative_correlations_lag_timeseries, lag_timeseries)
        self.assertEqual(self.negative_correlations_solution, correlations)

        self.assertIsInstance(corr, sacf.correlator.Correlator)

    def test_simple_correlation(self):
        lag_timeseries, correlations = SACF(
            self.timestamps1, self.data1
        ).autocorrelation(min_lag=0)

        self.assertEqual(self.simple_correlations_lag_timeseries, lag_timeseries)
        self.assertEqual(self.simple_correlation_solution, correlations)

    def test_two_correlation(self):
        (
            lag_timeseries,
            correlations,
        ) = SACF(self.timestamps1, [self.data1, self.data1]).autocorrelation(min_lag=0)

        self.assertEqual(self.simple_correlations_lag_timeseries, lag_timeseries)
        self.assertEqual(self.two_correlation_solution, correlations)

    def test_add_zero_lag(self):
        lag_timeseries, _ = SACF(self.timestamps1, self.data1).autocorrelation(
            lag_resolution=1.5
        )

        self.assertEqual(self.add_zero_lag_timeseries, lag_timeseries)

    def test_creation_from_file(self):
        sacf_instance = SACF(filename=os.path.join(THIS_DIR, "csv_with_titles.csv"))
        self.assertIsInstance(sacf_instance.data, sacf.datastructure.DataStructure)

    def test_specified_function_autocorrelation_gaussian(self):
        lag_timeseries, correlations, corr = SACF(
            self.timestamps1, self.data1
        ).autocorrelation(return_correlator=True, weight_function="gaussian")

        self.assertEqual(self.negative_correlations_lag_timeseries, lag_timeseries)
        self.assertEqual(self.negative_correlations_solution, correlations)

    def test_specified_function_autocorrelation_fractional_squared(self):
        lag_timeseries, correlations, corr = SACF(
            self.timestamps1, self.data1
        ).autocorrelation(return_correlator=True, weight_function="fractional_squared")

        self.assertEqual(self.negative_correlations_lag_timeseries, lag_timeseries)
        self.assertEqual(self.negative_correlations_solution, correlations)

    def test_specified_function_autocorrelation_fast(self):
        lag_timeseries, correlations, corr = SACF(
            self.timestamps1, self.data1
        ).autocorrelation(return_correlator=True, selection_function="fast")

        self.assertEqual(self.negative_correlations_lag_timeseries, lag_timeseries)
        self.assertNotEqual(
            self.negative_correlations_solution, correlations
        )  # fast indexing does not work...


if __name__ == "__main__":
    unittest.main()
