#include "JohansenHelper.h"

extern "C" {
    int coint_eigen(const double* doubleMat, int sampleCount, int seriesCount, int nlags, double* output_eig, double* output_stat, double* output_cvm) 
    {
        gsl_set_error_handler_off(); // Disable default GSL error handler

        int nc = seriesCount;
        int nr = sampleCount;
        gsl_matrix* xMat_gsl = gsl_matrix_alloc(nr, nc);

        for (int i = 0; i < nc; i++)
        {
            for (int j = 0; j < nr; j++)
            {
                gsl_matrix_set(xMat_gsl, j, i, doubleMat[j*seriesCount + i]);
                // printf("[%d, %d]: %f\n", j, i, doubleMat[j*seriesCount + i]);
            }
        }

        JohansenHelper johansenHelper(make_shared_matrix(xMat_gsl));
        johansenHelper.DoMaxEigenValueTest(nlags);

        const vector<MaxEigenData> &outStats = johansenHelper.GetOutStats();
        for (int i = 0; i < (int)outStats.size(); i++)
        {
            output_stat[i] = outStats[i].TestStatistic;
            output_cvm[i * 3 + 0] = outStats[i].CriticalValue90;
            output_cvm[i * 3 + 1] = outStats[i].CriticalValue95;
            output_cvm[i * 3 + 2] = outStats[i].CriticalValue99;
            // printf("%d Test Stat: %f 90: %f 95: %f 99: %f\n",
            //     i, outStats[i].TestStatistic, outStats[i].CriticalValue90, outStats[i].CriticalValue95, outStats[i].CriticalValue99);
        }

        const DoubleVector &eigenValuesVec = johansenHelper.GetEigenValues();
        for (int i = 0; i < (int)eigenValuesVec.size(); i++)
        {
            output_eig[i] = eigenValuesVec[i];
        }

        int cointCount = johansenHelper.CointegrationCount();
        return cointCount;
    }


    void coint_rolling(const double* doubleMat, int sampleCount, int seriesCount, int nlags, int window, int* output_count)
    {
        gsl_set_error_handler_off(); // Disable default GSL error handler

        int nc = seriesCount;
        int nr = sampleCount;
        gsl_matrix* xMat_gsl = gsl_matrix_alloc(nr, nc);

        for (int i = 0; i < nc; i++)
        {
            for (int j = 0; j < nr; j++)
            {
                gsl_matrix_set(xMat_gsl, j, i, doubleMat[j*seriesCount + i]);
                // printf("[%d, %d]: %f\n", j, i, doubleMat[j*seriesCount + i]);
            }
        }

        shared_matrix xMat = make_shared_matrix(xMat_gsl);

        for (int i = window - 1; i < nr; i++)
        {
            shared_matrix subMax = GetSubMatrix(xMat, i - window + 1, i, -1, -1);
            JohansenHelper johansenHelper(subMax);
            johansenHelper.DoMaxEigenValueTest(nlags);
            output_count[i] = johansenHelper.CointegrationCount();
        }
    }
}

/*
$ sudo apt-get install libgsl-dev
$ g++ -std=c++17 -o JohansenTest JohansenTest.cpp JohansenHelper.cpp -lgsl -lgslcblas -lm 
$ g++ -std=c++17 -o JohansenTest_C.so JohansenHelper.cpp JohansenTest_C.cpp -lgsl -lgslcblas -lm -shared -fPIC -Wl,--no-undefined

https://blog.quantinsti.com/johansen-test-cointegration-building-stationary-portfolio/
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
data = np.vstack([np.loadtxt('gld.csv'), np.loadtxt('slv.csv')]).T
for nlags in range(1, 10):
    result = coint_johansen(data, det_order=0, k_ar_diff=nlags)
    print(nlags, result.cvm, result.trace_stat, result.trace_stat_crit_vals)

$ ./JohansenTest  # results between C++ and Python are close but not identical
[1] 0 Test Stat: 8.277511 90: 12.297100 95: 14.263900 99: 18.520000
[1] 1 Test Stat: 0.174714 90: 2.705500 95: 3.841500 99: 6.634900
[2] 0 Test Stat: 8.304559 90: 12.297100 95: 14.263900 99: 18.520000
[2] 1 Test Stat: 0.220921 90: 2.705500 95: 3.841500 99: 6.634900
[3] 0 Test Stat: 9.367500 90: 12.297100 95: 14.263900 99: 18.520000
[3] 1 Test Stat: 0.196428 90: 2.705500 95: 3.841500 99: 6.634900
[4] 0 Test Stat: 9.495959 90: 12.297100 95: 14.263900 99: 18.520000
[4] 1 Test Stat: 0.207145 90: 2.705500 95: 3.841500 99: 6.634900
[5] 0 Test Stat: 9.245345 90: 12.297100 95: 14.263900 99: 18.520000
[5] 1 Test Stat: 0.198913 90: 2.705500 95: 3.841500 99: 6.634900
[6] 0 Test Stat: 9.379770 90: 12.297100 95: 14.263900 99: 18.520000
[6] 1 Test Stat: 0.171199 90: 2.705500 95: 3.841500 99: 6.634900
[7] 0 Test Stat: 10.312501 90: 12.297100 95: 14.263900 99: 18.520000
[7] 1 Test Stat: 0.139826 90: 2.705500 95: 3.841500 99: 6.634900
[8] 0 Test Stat: 11.088901 90: 12.297100 95: 14.263900 99: 18.520000
[8] 1 Test Stat: 0.162717 90: 2.705500 95: 3.841500 99: 6.634900
[9] 0 Test Stat: 9.875771 90: 12.297100 95: 14.263900 99: 18.520000
[9] 1 Test Stat: 0.182212 90: 2.705500 95: 3.841500 99: 6.634900

$ ldd ./JohansenTest_C.so
$ nm -D ./JohansenTest_C.so  | grep gsl_matrix_set_col

$ LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu python
import numpy as np
from ctypes import CDLL, c_double, POINTER, c_int, util
from numba import njit, jit

# Load the compiled shared library
lib = CDLL('./JohansenTest_C.so')

# Bind the C++ function
coint_eigen_c = lib.coint_eigen
coint_eigen_c.argtypes = [POINTER(c_double), c_int, c_int, c_int, POINTER(c_double)]
coint_eigen_c.restype = c_int

def call_cpp_coint_eigen(input_matrix, nlags):
    """
    Call the C++ function to flatten the 2D array.
    """
    rows, cols = input_matrix.shape
    output_eig = np.empty(cols, dtype=np.float64)
    output_stat = np.empty(cols, dtype=np.float64)
    output_cvm = np.empty(cols * 3, dtype=np.float64)
    # Call the C++ function
    cnt = coint_eigen_c(
        input_matrix.ctypes.data_as(POINTER(c_double)),
        c_int(rows),
        c_int(cols),
        c_int(nlags),
        output_eig.ctypes.data_as(POINTER(c_double)),
        output_stat.ctypes.data_as(POINTER(c_double)),
        output_cvm.ctypes.data_as(POINTER(c_double)),
    )
    return cnt, output_eig, output_stat, output_cvm.reshape((cols, 3))

data = np.random.rand(20, 4)
cnt, output_eig, output_stat, output_cvm = call_cpp_coint_eigen(data, 1)
from statsmodels.tsa.vector_ar.vecm import coint_johansen
result = coint_johansen(data, det_order=0, k_ar_diff=1)
print(result.cvm, result.trace_stat, result.trace_stat_crit_vals)
np.testing.assert_allclose(result.eig, output_eig)
np.testing.assert_allclose(result.cvm, output_cvm)
print(output_stat, result.trace_stat)
*/