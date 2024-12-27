#include "JohansenHelper.h"
#include <gsl/gsl_eigen.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>

using namespace std;
using namespace CommonTypes;

namespace
{
    double maxEigenCriticalvalues[][3] =
        {
        {2.7055,    3.8415,     6.6349},
        {12.2971,   14.2639,    18.52},
        {18.8928,   21.1314,    25.865},
        {25.1236,   27.5858,    32.7172},
        {31.2379,   33.8777,    39.3693},
        {37.2786,   40.0763,    45.8662},
        {43.2947,   46.2299,    52.3069},
        {49.2855,   52.3622,    58.6634},
        {55.2412,   58.4332,    64.996},
        {61.2041,   64.504,     71.2525},
        {67.1307,   70.5392,    77.4877},
        {73.0563,   76.5734,    83.7105}
        };
}

shared_matrix make_shared_matrix(gsl_matrix* x) 
{
    return shared_matrix(x, [](gsl_matrix* m) {gsl_matrix_free(m);});
}

shared_matrix GetSubMatrix(shared_matrix xMat, int nBeginRow, int nEndRow, int nBeginCol, int nEndCol)
{
    int nr = xMat->size1;
    int nc = xMat->size2;

    if (nBeginRow == -1) nBeginRow = 0;
    if (nEndRow == -1) nEndRow = nr - 1;
    if (nBeginCol == -1) nBeginCol = 0;
    if (nEndCol == -1) nEndCol = nc - 1;

    int newRows = nEndRow - nBeginRow + 1;
    int newColumns = nEndCol - nBeginCol + 1;

    gsl_matrix_view subMatrix = gsl_matrix_submatrix(xMat.get(), nBeginRow, nBeginCol, newRows, newColumns);
    gsl_matrix* retVal = gsl_matrix_alloc(subMatrix.matrix.size1, subMatrix.matrix.size2);
    gsl_matrix_memcpy(retVal, &subMatrix.matrix);

    return make_shared_matrix(retVal);
}

shared_matrix GetMatrixDifference(shared_matrix xMat)
{
    int nr = xMat->size1;
    int nc = xMat->size2;

    gsl_matrix* diffMatrix = gsl_matrix_alloc(nr-1,nc);

    for (int i = 0; i < nc; i++)
    {
        for (int j = 0; j < nr - 1; j++)
        {
            double diff = gsl_matrix_get(xMat.get(), j + 1, i) - gsl_matrix_get(xMat.get(), j, i);
            gsl_matrix_set(diffMatrix, j, i, diff);
        }
    }

    return make_shared_matrix(diffMatrix);
}

double GetAverage(gsl_vector* x);

shared_matrix DeMean(shared_matrix xMat)
{
    int nr = xMat->size1;
    int nc = xMat->size2;
    gsl_matrix* retVal = gsl_matrix_alloc(nr,nc);

    for (int i = 0; i < nc; i++)
    {
        gsl_vector* currentCol = gsl_vector_alloc(nr);
        gsl_matrix_get_col(currentCol, xMat.get(), i);

        double average = GetAverage(currentCol);
        for (int j = 0; j < nr; j++)
        {
            double demeanedValue = gsl_vector_get(currentCol, j) -average;
            gsl_vector_set(currentCol, j, demeanedValue);
        }
        gsl_matrix_set_col(retVal, i, currentCol);
        gsl_vector_free(currentCol);
    }

    return make_shared_matrix(retVal);
}

shared_matrix GetMatrixLagged(shared_matrix xMat, int nlags)
{
    int nr = xMat->size1;
    int nc = xMat->size2;

    int nRows = nr - nlags;
    int nCols = nc * nlags;
    gsl_matrix* retVal = gsl_matrix_alloc(nRows, nCols);

    int counter = 0;
    int counter2 = 0;
    for (int i = 0; i < nCols; i++)
    {
        for (int j = 0; j < nRows; j++)
        {
            int laggedRow = j + nlags - counter - 1;
            double laggedValue = gsl_matrix_get(xMat.get(), laggedRow, counter2);
            gsl_matrix_set(retVal, j, i, laggedValue);
        }
        counter++;
        if (counter >= nlags)
        {
            counter = 0;
            counter2++;
        }
    }

    return make_shared_matrix(retVal);
}

double GetAverage(gsl_vector* x)
{
    double retVal = 0;
    int n = x->size;

    for (int i = 0; i < n; i++)
    {
        double element = gsl_vector_get(x, i);
        retVal += element;
    }

    retVal = retVal / n;

    return retVal;
}

void MatrixDivideByElem(shared_matrix xMat, double val)
{
    size_t nr = xMat->size1;
    size_t nc = xMat->size2;
    gsl_matrix* tmpMat = gsl_matrix_alloc(nr, nc);
    gsl_matrix_set_all(tmpMat, val);
    gsl_matrix_div_elements(xMat.get(), tmpMat);
    gsl_matrix_free(tmpMat);
}

shared_matrix MatrixTransposeImpl(shared_matrix m)
{
    int nr = m->size1;
    int nc = m->size2;

    gsl_matrix* retmat = gsl_matrix_alloc(nc, nr);
    gsl_matrix_transpose_memcpy(retmat, m.get());

    return make_shared_matrix(retmat);
}

shared_matrix MatrixMultiply(shared_matrix A, shared_matrix B)
{
    int nrA = A->size1;
    int ncB = B->size2;

    gsl_matrix* resMat = gsl_matrix_alloc(nrA, ncB);
    gsl_blas_dgemm(CblasNoTrans,
        CblasNoTrans,
        1.0, A.get(), B.get(),
        0.0, resMat);

    return make_shared_matrix(resMat);
}

shared_matrix GetMatrixInverse(shared_matrix inMat)
{
    int size1 = inMat->size1;
    gsl_matrix* outMat = gsl_matrix_alloc(size1, size1);
    gsl_matrix* invert_me = gsl_matrix_alloc(size1, size1);
    gsl_permutation* perm = gsl_permutation_alloc(size1);
    int signum;
    gsl_matrix_memcpy(invert_me, inMat.get());
    gsl_linalg_LU_decomp(invert_me, perm, &signum);
    gsl_linalg_LU_invert(invert_me, perm, outMat);
    gsl_matrix_free(invert_me);
    gsl_permutation_free(perm);

    return make_shared_matrix(outMat);
}

shared_matrix MatrixDivide(shared_matrix xMat, shared_matrix yMat)
{
    //Dim xTranspose As Variant
    shared_matrix xTranspose = MatrixTransposeImpl(xMat);
    shared_matrix tmp1 = MatrixMultiply(xTranspose, xMat);
    shared_matrix tmp2 = GetMatrixInverse(tmp1);
    shared_matrix tmp3 = MatrixMultiply(tmp2, xTranspose);
    shared_matrix tmp4 = MatrixMultiply(tmp3, yMat);
    return tmp4;
}

shared_matrix DoubleMatrixToGSLMatrix(const DoubleMatrix &doubleMat)
{
    int nc = doubleMat.size();
    int nr = doubleMat[0].size();
    gsl_matrix* x = gsl_matrix_alloc(nr, nc);

    for (int i = 0; i < nc; i++)
    {
        for (int j = 0; j < nr; j++)
        {
            gsl_matrix_set(x, j, i, doubleMat[i][j]);
        }
    }

    return make_shared_matrix(x);
}

DoubleVector GSLComplexVecToAbsDoubleVector(gsl_vector_complex* gslVec)
{
    int n = gslVec->size;
    DoubleVector retvec(n);
    for (int i = 0; i < n; i++)
    {
        gsl_complex matComp = gsl_vector_complex_get(gslVec, i);
        double realval = matComp.dat[0];
        double imagval = matComp.dat[1];
        retvec[i] = realval;
    }

    return retvec;
}

DoubleMatrix GSLComplexMatToAbsDoubleMatrix(gsl_matrix_complex* gslMat)
{
    int nr = gslMat->size1;
    int nc = gslMat->size2;
    DoubleMatrix retmat(nr);
    for (int i = 0; i < nr; i++)
    {
        retmat[i].resize(nc);
        for (int j = 0; j < nc; j++)
        {
            gsl_complex matComp = gsl_matrix_complex_get(gslMat, i, j);
            double realval = matComp.dat[0];
            double imagval = matComp.dat[1];

            retmat[i][j] = realval;
        }
    }

    return retmat;
}

JohansenHelper::JohansenHelper(const DoubleMatrix &xMat)
{
    // Convert input matrix to gsl_matrix
    this->xMat_gsl = DoubleMatrixToGSLMatrix(xMat);
}

int JohansenHelper::CointegrationCount() const
{
    int retVal = 0;
    int foundCointegrations = 0;
    bool conclusive = false;
    for (int i = 0; i < (int)this->outStats.size(); i++)
    {
        if (conclusive == false)
        {
            if ((this->outStats[i].TestStatistic < this->outStats[i].CriticalValue90) &&
                (this->outStats[i].TestStatistic < this->outStats[i].CriticalValue95) &&
                (this->outStats[i].TestStatistic < this->outStats[i].CriticalValue99))
            {
                foundCointegrations = i;
                conclusive = true;
            }
        }
    }

    if (conclusive == false)
    {
        retVal = -1;
    }
    else
    {
        retVal = foundCointegrations;
    }

    return retVal;
}

void JohansenHelper::DoMaxEigenValueTest(int nlags)
{
    // Demean input data in place
    shared_matrix xMat_temp = DeMean(this->xMat_gsl);
    gsl_matrix_memcpy(this->xMat_gsl.get(), xMat_temp.get());

    // Get inter-sample differences
    shared_matrix dxMat_gsl = GetMatrixDifference(this->xMat_gsl);
        
    // Demean the lagged differenced data
    shared_matrix dxLaggedDemeanedMatrix_gsl = DeMean(GetMatrixLagged(dxMat_gsl, nlags));

    // Pull out the difference data excluding the lagged samples, then demean them
    shared_matrix dxDemeanedMatrix_gsl = DeMean(GetSubMatrix(dxMat_gsl, nlags, -1, -1, -1));

    int nrx = dxLaggedDemeanedMatrix_gsl->size1;
    int ncx = dxLaggedDemeanedMatrix_gsl->size2;
    int nry = dxDemeanedMatrix_gsl->size1;
    int ncy = dxDemeanedMatrix_gsl->size2;

    // Mat divide the lagged diff data by the differenced data
    shared_matrix tmp1 = MatrixDivide(dxLaggedDemeanedMatrix_gsl, dxDemeanedMatrix_gsl);

    // Mat multiply the lagged diff data by the differenced data to yield the fitted regression
    shared_matrix fittedRegressionDX = MatrixMultiply(dxLaggedDemeanedMatrix_gsl, tmp1);

    int nr = dxDemeanedMatrix_gsl->size1;
    int nc = dxDemeanedMatrix_gsl->size2;

    // Subtract the fitted regression from the residu
    shared_matrix ResidualsRegressionDX = make_shared_matrix(gsl_matrix_alloc(nr, nc));
    gsl_matrix_memcpy(ResidualsRegressionDX.get(), dxDemeanedMatrix_gsl.get());
    gsl_matrix_sub(ResidualsRegressionDX.get(), fittedRegressionDX.get());

    nrx = this->xMat_gsl->size1;
    shared_matrix tmp4_gsl =GetSubMatrix(this->xMat_gsl, 1, nrx - nlags - 1, -1, -1);
    shared_matrix xDemeanedMatrix_gsl = DeMean(tmp4_gsl);

    shared_matrix tmp6 = MatrixDivide(dxLaggedDemeanedMatrix_gsl, xDemeanedMatrix_gsl);
    shared_matrix fittedRegressionX = MatrixMultiply(dxLaggedDemeanedMatrix_gsl, tmp6);

    shared_matrix ResidualsRegressionX = make_shared_matrix(gsl_matrix_alloc(xDemeanedMatrix_gsl->size1, xDemeanedMatrix_gsl->size2));
    gsl_matrix_memcpy(ResidualsRegressionX.get(), xDemeanedMatrix_gsl.get());
    gsl_matrix_sub(ResidualsRegressionX.get(), fittedRegressionX.get());

    shared_matrix tposeResid = MatrixTransposeImpl(ResidualsRegressionX);
    shared_matrix Skk = MatrixMultiply(tposeResid, ResidualsRegressionX);
    MatrixDivideByElem(Skk, (double)ResidualsRegressionX->size1);

    shared_matrix Sk0 = MatrixMultiply(tposeResid, ResidualsRegressionDX);
    MatrixDivideByElem(Sk0, (double)ResidualsRegressionX->size1);

    tposeResid = MatrixTransposeImpl(ResidualsRegressionDX);
    shared_matrix S00 = MatrixMultiply(tposeResid, ResidualsRegressionDX);
    MatrixDivideByElem(S00, (double)ResidualsRegressionDX->size1);

    tposeResid = MatrixTransposeImpl(Sk0);
    shared_matrix skkInverse = GetMatrixInverse(Skk);
    shared_matrix s00Inverse = GetMatrixInverse(S00);
    shared_matrix matMultTemp1 = MatrixMultiply(Sk0, s00Inverse);
    shared_matrix matMultTemp2 = MatrixMultiply(matMultTemp1, tposeResid);
    shared_matrix eigenInputMat = MatrixMultiply(skkInverse, matMultTemp2);

    int n = eigenInputMat->size1;

    gsl_vector_complex* evalPtr = gsl_vector_complex_alloc(n);
    gsl_matrix_complex* ematPtr = gsl_matrix_complex_alloc(n, n);
    gsl_eigen_nonsymmv_workspace* worspacePtr = gsl_eigen_nonsymmv_alloc(n);
    gsl_eigen_nonsymmv(eigenInputMat.get(), evalPtr, ematPtr, worspacePtr);
    gsl_eigen_nonsymmv_free(worspacePtr);

    gsl_eigen_nonsymmv_sort(evalPtr, ematPtr, GSL_EIGEN_SORT_ABS_DESC);

    this->eigenValuesVec = GSLComplexVecToAbsDoubleVector(evalPtr);
    this->eigenVecMatrix = GSLComplexMatToAbsDoubleMatrix(ematPtr);

    gsl_vector_complex_free(evalPtr);
    gsl_matrix_complex_free(ematPtr);

    int nSamples = ResidualsRegressionX->size1;
    int nVariables = ResidualsRegressionX->size2;

    int counter = 0;
    this->outStats.clear();
    for (int i = 0; i < (int)this->eigenValuesVec.size(); i++)
    {
        MaxEigenData eigData;

        double LR_maxeigenvalue = -nSamples * log(1 - this->eigenValuesVec[i]);
        eigData.TestStatistic = LR_maxeigenvalue;
        eigData.CriticalValue90 = maxEigenCriticalvalues[nVariables - counter - 1][0];
        eigData.CriticalValue95 = maxEigenCriticalvalues[nVariables - counter - 1][1];
        eigData.CriticalValue99 = maxEigenCriticalvalues[nVariables - counter - 1][2];
        counter++;
        this->outStats.push_back(eigData);
    }
}
