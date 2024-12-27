#ifndef JOHANSEN_HELPER_DEFINED
#define JOHANSEN_HELPER_DEFINED

#include "CommonTypes.h"
#include <gsl/gsl_matrix.h>
#include <memory>

using namespace std;
using namespace CommonTypes;

typedef shared_ptr<gsl_matrix> shared_matrix;
shared_matrix make_shared_matrix(gsl_matrix* x);

class JohansenHelper
{
public:
    JohansenHelper(const JohansenHelper &x) {*this = x;};
    JohansenHelper(const DoubleMatrix &xMat);
    JohansenHelper(shared_matrix xMat) {this->xMat_gsl = xMat;}

    void DoMaxEigenValueTest(int nlags);
    int CointegrationCount() const;
    const vector<MaxEigenData> &GetOutStats() const {return this->outStats;}
    const DoubleVector &GetEigenValues() const {return this->eigenValuesVec;}
    const DoubleMatrix &GetEigenVecMatrix() const {return this->eigenVecMatrix;}

private:
    // Data members
    shared_matrix xMat_gsl;
    vector<MaxEigenData> outStats;
    DoubleVector eigenValuesVec;
    DoubleMatrix eigenVecMatrix;
};

#endif
