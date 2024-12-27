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

