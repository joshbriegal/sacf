import unittest
import GACF
import GACF.datastructure
import GACF.correlator
import numpy as np
import os

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
    float_list = [0., 1., 2., 3., 4.]
    float_list_mean = 2.
    float_list_median = 2.
    numpy_float_array = np.array(float_list)
    empty_list = []
    empty_numpy_array = np.array([])
    numpy_float_array_with_nan = np.append(numpy_float_array, np.nan)
    numpy_float_array_with_nan_median = 2.5

    # tests for 2 list constructor

    def test_import_integer_lists(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.integer_list, TestDataStructure.integer_list)

    def test_import_float_lists(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.float_list, TestDataStructure.float_list)

    def test_import_numpy_float_array(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.numpy_float_array, TestDataStructure.numpy_float_array)

    def test_import_empty_list(self):
        with self.assertRaises(GACF.datastructure.EmptyDataStructureException):
            ds = GACF.datastructure.DataStructure(TestDataStructure.empty_list, TestDataStructure.empty_list)

    def test_import_empty_numpy_array(self):
        with self.assertRaises(GACF.datastructure.EmptyDataStructureException):
            ds = GACF.datastructure.DataStructure(TestDataStructure.empty_numpy_array,
                                                  TestDataStructure.empty_numpy_array)

    def test_import_numpy_float_array_with_nan(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.numpy_float_array_with_nan,
                                              TestDataStructure.numpy_float_array_with_nan)

    def test_uneven_lists(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.short_integer_list, TestDataStructure.integer_list)

    def test_uneven_lists2(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.integer_list, TestDataStructure.short_integer_list)

    # Tests for 3 list constructor

    def test_import_integer_lists_with_err(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.integer_list, TestDataStructure.integer_list,
                                              TestDataStructure.integer_list)

    def test_import_float_lists_with_err(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.float_list, TestDataStructure.float_list,
                                              TestDataStructure.float_list)

    def test_import_numpy_float_array_with_err(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.numpy_float_array, TestDataStructure.numpy_float_array,
                                              TestDataStructure.numpy_float_array)

    def test_import_empty_list_with_err(self):
        with self.assertRaises(GACF.datastructure.EmptyDataStructureException):
            ds = GACF.datastructure.DataStructure(TestDataStructure.empty_list, TestDataStructure.empty_list,
                                                  TestDataStructure.empty_list)

    def test_import_empty_numpy_array_with_err(self):
        with self.assertRaises(GACF.datastructure.EmptyDataStructureException):
            ds = GACF.datastructure.DataStructure(TestDataStructure.empty_numpy_array,
                                                  TestDataStructure.empty_numpy_array,
                                                  TestDataStructure.empty_numpy_array)

    def test_import_numpy_float_array_with_nan_with_err(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.numpy_float_array_with_nan,
                                              TestDataStructure.numpy_float_array_with_nan,
                                              TestDataStructure.numpy_float_array_with_nan)

    # Tests for file based constructor

    def test_import_csv_with_title(self):
        ds = GACF.datastructure.DataStructure(os.path.join(THIS_DIR, 'csv_with_titles.csv'))

    def test_import_csv_with_title_commented(self):
        ds = GACF.datastructure.DataStructure(os.path.join(THIS_DIR, 'csv_with_titles_commented.csv'))

    def test_import_csv(self):
        ds = GACF.datastructure.DataStructure(os.path.join(THIS_DIR, 'csv_no_titles.csv'))

    def test_import_csv_with_err(self):
        ds = GACF.datastructure.DataStructure(os.path.join(THIS_DIR, 'csv_no_titles_with_err.csv'))

    def test_import_csv_with_err_with_title(self):
        ds = GACF.datastructure.DataStructure(os.path.join(THIS_DIR, 'csv_with_titles_with_err.csv'))

    def test_import_tab_delim_file_with_title(self):
        ds = GACF.datastructure.DataStructure(os.path.join(THIS_DIR, 'tab_delim_with_titles.txt'))

    def test_import_tab_delim_file_with_title_commented(self):
        ds = GACF.datastructure.DataStructure(os.path.join(THIS_DIR, 'tab_delim_with_titles_commented.txt'))

    def test_import_tab_delim_file(self):
        ds = GACF.datastructure.DataStructure(os.path.join(THIS_DIR, 'tab_delim_no_titles.txt'))

    def test_import_tab_delim_file_with_err(self):
        ds = GACF.datastructure.DataStructure(os.path.join(THIS_DIR, 'tab_delim_no_titles_with_err.txt'))

    def test_import_tab_delim_file_with_err_with_title(self):
        ds = GACF.datastructure.DataStructure(os.path.join(THIS_DIR, 'tab_delim_with_titles_with_err.txt'))

    # method & property tests

    def test_data(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.integer_list, TestDataStructure.integer_list2)
        self.assertEqual(TestDataStructure.integer_list2, ds.data()[0])

    def test_data_2d(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.integer_list,
                                              [TestDataStructure.integer_list2, TestDataStructure.integer_list])
        self.assertEqual([TestDataStructure.integer_list2, TestDataStructure.integer_list], ds.data())

    def test_timeseries(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.integer_list2, TestDataStructure.integer_list)
        self.assertEqual(TestDataStructure.integer_list2, ds.timeseries())

    def test_errors_empty(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.integer_list2, TestDataStructure.integer_list)
        self.assertEqual(TestDataStructure.empty_list, ds.errors())

    def test_errors(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.integer_list, TestDataStructure.integer_list,
                                              TestDataStructure.integer_list2)
        self.assertEqual(TestDataStructure.integer_list2, ds.errors()[0])

    def test_normalised_data(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.integer_list, TestDataStructure.integer_list2)
        self.assertEqual([x - TestDataStructure.integer_list2_mean for x in TestDataStructure.integer_list2],
                         ds.normalised_data()[0])

    def test_normalised_timeseries(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.integer_list2, TestDataStructure.integer_list)
        self.assertEqual(TestDataStructure.integer_list, ds.normalised_timeseries())

    def test_mean_data(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.integer_list, TestDataStructure.integer_list2)
        self.assertEqual(TestDataStructure.integer_list2_mean, ds.mean_data[0])

    def test_mean_data_nan(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.integer_list,
                                              TestDataStructure.numpy_float_array_with_nan)
        self.assertEqual(TestDataStructure.float_list_mean, ds.mean_data[0])

    def test_median_time(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.float_list,
                                              TestDataStructure.integer_list)
        self.assertEqual(TestDataStructure.float_list_median, ds.median_time)

    def test_median_time_nan(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.numpy_float_array_with_nan,
                                              TestDataStructure.integer_list)
        self.assertEqual(TestDataStructure.numpy_float_array_with_nan_median, ds.median_time)

    def test_max_time(self):
        ds = GACF.datastructure.DataStructure(TestDataStructure.float_list,
                                              TestDataStructure.integer_list)
        self.assertEqual(TestDataStructure.float_list[-1], ds.max_time)


class TestCorrelator(unittest.TestCase):

    def setUp(self):
        self.timestamps1 = [0, 1, 2, 3, 4]
        self.timestamps1_median = 2
        self.data1 = [0, 1, 0, -1, 0]
        self.data2 = [0, 1, 2, 1, np.nan]
        self.ds1 = GACF.datastructure.DataStructure(self.timestamps1, self.data1)
        self.ds2 = GACF.datastructure.DataStructure(self.timestamps1, [self.data1, self.data2])
        self.simple_correlation_solution = {'lag_timeseries': [0., 1., 2., 3., 4.],
                                            'correlations': [1.0, 0.0, -0.5, 0.0, 0.0]}
        self.two_correlation_solution = {'lag_timeseries': [0., 1., 2., 3., 4.],
                                         'correlations': [[1.0, 0.0, -0.5, 0.0, 0.0],
                                                          [1.0, 0.0, -0.5, 0.0, 0.0]]}

    def test_setup_correlator(self):
        corr = GACF.correlator.Correlator(self.ds1)

    def test_setup_correlationiterator(self):
        cor_it = GACF.correlator.CorrelationIterator(1, 1)

    def test_default_max_lag(self):
        corr = GACF.correlator.Correlator(self.ds1)
        self.assertEqual(self.timestamps1[-1], corr.max_lag)

    def test_alter_max_lag(self):
        max_lag = 100
        corr = GACF.correlator.Correlator(self.ds1)
        corr.max_lag = max_lag
        self.assertEqual(max_lag, corr.max_lag)

    def test_default_lag_resolution(self):
        corr = GACF.correlator.Correlator(self.ds1)
        self.assertEqual(float(self.timestamps1[-1]) / float(len(self.timestamps1)), corr.lag_resolution)

    def test_alter_lag_resolution(self):
        lag_res = 100
        corr = GACF.correlator.Correlator(self.ds1)
        corr.lag_resolution = 100
        self.assertEqual(lag_res, corr.lag_resolution)

    def test_default_alpha(self):
        corr = GACF.correlator.Correlator(self.ds1)
        self.assertEqual(self.timestamps1_median, corr.alpha)

    def test_alter_alpha(self):
        alpha = 100
        corr = GACF.correlator.Correlator(self.ds1)
        corr.alpha = 100
        self.assertEqual(alpha, corr.alpha)

    def test_simple_correlation(self):
        correlations, _ = GACF.find_correlation_from_lists(self.timestamps1, self.data1, lag_resolution=1.0)
        self.assertEqual(correlations, self.simple_correlation_solution)

    def test_two_correlation(self):
        correlations, _ = GACF.find_correlation_from_lists(self.timestamps1, [self.data1, self.data1],
                                                           lag_resolution=1.0)
        self.assertEqual(correlations, self.two_correlation_solution)


if __name__ == '__main__':
    unittest.main()
