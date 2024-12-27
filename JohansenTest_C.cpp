#include "JohansenHelper.h"

extern "C" {
    int coint(double* doubleMat, int sampleCount, int seriesCount, int nlags) {
        int nc = seriesCount;
        int nr = sampleCount;
        gsl_matrix* xMat = gsl_matrix_alloc(nr, nc);

        for (int i = 0; i < nc; i++)
        {
            for (int j = 0; j < nr; j++)
            {
                gsl_matrix_set(xMat, j, i, doubleMat[i*seriesCount + j]);
            }
        }

        JohansenHelper johansenHelper(make_shared_matrix(xMat));
        johansenHelper.DoMaxEigenValueTest(nlags);

        const vector<MaxEigenData> &outStats = johansenHelper.GetOutStats();
        for (int i = 0; i < (int)outStats.size(); i++)
        {
            printf("%d Test Stat: %f 90: %f 95: %f 99: %f\n",
                i, outStats[i].TestStatistic, outStats[i].CriticalValue90, outStats[i].CriticalValue95, outStats[i].CriticalValue99);
        }

        int cointCount = johansenHelper.CointegrationCount();
        return cointCount;
    }
}

// $ sudo apt-get install libgsl-dev
// $ g++ -std=c++17 -lgsl -lgslcblas -lm -o JohansenTest JohansenTest.cpp JohansenHelper.cpp
// $ g++ -std=c++17 -lgsl -lgslcblas -lm -shared -fPIC -o JohansenTest_C.so JohansenHelper.cpp JohansenTest_C.cpp

// https://blog.quantinsti.com/johansen-test-cointegration-building-stationary-portfolio/
// from statsmodels.tsa.vector_ar.vecm import coint_johansen
// import numpy as np
// data = np.vstack([np.loadtxt('gld.csv'), np.loadtxt('slv.csv')]).T
// for nlags in range(1, 10):
//     result = coint_johansen(data, det_order=0, k_ar_diff=nlags)
//     print(nlags, result.cvm, result.trace_stat, result.trace_stat_crit_vals)

// ./JohansenTest  # results between C++ and Python are close but not identical
// [1] 0 Test Stat: 8.277511 90: 12.297100 95: 14.263900 99: 18.520000
// [1] 1 Test Stat: 0.174714 90: 2.705500 95: 3.841500 99: 6.634900
// [2] 0 Test Stat: 8.304559 90: 12.297100 95: 14.263900 99: 18.520000
// [2] 1 Test Stat: 0.220921 90: 2.705500 95: 3.841500 99: 6.634900
// [3] 0 Test Stat: 9.367500 90: 12.297100 95: 14.263900 99: 18.520000
// [3] 1 Test Stat: 0.196428 90: 2.705500 95: 3.841500 99: 6.634900
// [4] 0 Test Stat: 9.495959 90: 12.297100 95: 14.263900 99: 18.520000
// [4] 1 Test Stat: 0.207145 90: 2.705500 95: 3.841500 99: 6.634900
// [5] 0 Test Stat: 9.245345 90: 12.297100 95: 14.263900 99: 18.520000
// [5] 1 Test Stat: 0.198913 90: 2.705500 95: 3.841500 99: 6.634900
// [6] 0 Test Stat: 9.379770 90: 12.297100 95: 14.263900 99: 18.520000
// [6] 1 Test Stat: 0.171199 90: 2.705500 95: 3.841500 99: 6.634900
// [7] 0 Test Stat: 10.312501 90: 12.297100 95: 14.263900 99: 18.520000
// [7] 1 Test Stat: 0.139826 90: 2.705500 95: 3.841500 99: 6.634900
// [8] 0 Test Stat: 11.088901 90: 12.297100 95: 14.263900 99: 18.520000
// [8] 1 Test Stat: 0.162717 90: 2.705500 95: 3.841500 99: 6.634900
// [9] 0 Test Stat: 9.875771 90: 12.297100 95: 14.263900 99: 18.520000
// [9] 1 Test Stat: 0.182212 90: 2.705500 95: 3.841500 99: 6.634900
