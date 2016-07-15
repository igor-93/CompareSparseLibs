#include <vector>
#include <iostream>
#include <ctime>
#include <stdlib.h>

#include <Eigen/Sparse>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_spblas.h>

//#include "SparseMatrix.h"

using namespace std;
using namespace Eigen;

int main (int argc, char** argv) {

    size_t mat_size = 10000;
    size_t iters = 1;

    // Initialize matrices
    SparseMatrix<double> matEigen(mat_size,mat_size);
    gsl_spmatrix *matGSL;
    matGSL = gsl_spmatrix_alloc(mat_size,mat_size);

    cout << "Filling the matrices... this may take a while..." << endl;
    srand (time(NULL));
    int count = 0;
    for(int i = 0; i < mat_size; i++){
        for(int j = 0; j < mat_size; j++){
            const double f = (double)rand() / RAND_MAX;
            if(f < 0.1){
                count++;
                matEigen.insert(i, j) = f;
                gsl_spmatrix_set(matGSL, i, j, f);
            }
        }
    }

    cout << "We have in total " << 100*(double)(count) / (double)(mat_size*mat_size) << "% entries in the matrices." << endl;
    // Convert the gsl matrix from triplet to col compact format
    matGSL = gsl_spmatrix_compcol(matGSL);
    gsl_spmatrix *C = gsl_spmatrix_alloc(mat_size,mat_size);
    gsl_spmatrix_set_zero(C);
    C = gsl_spmatrix_compcol(C);
    cout << "Starting multiplication with GSL..." << endl;
    clock_t begin = clock();
    double alpha = 1;
    for(int i = 0; i < iters; i ++) {
        gsl_spblas_dgemm(alpha, matGSL, matGSL, C);
        gsl_spmatrix_memcpy(matGSL, C);
    }
    clock_t mid = clock();

    cout << "Starting multiplication with Eigen..." << endl;
    for(int i = 0; i < iters; i++) {
        matEigen = matEigen * matEigen;
    }
    clock_t end = clock();

    double elapsed_gsl = double(mid - begin) / CLOCKS_PER_SEC;
    double elapsed_eigen = double(end - mid) / CLOCKS_PER_SEC;

    cout << "Elapsed time GSL: " << elapsed_gsl << endl;
    cout << "Elapsed time Eigen: " << elapsed_eigen << endl;

    // Print the result to make sure that GSL is implemented correctly
    cout << "NZ: (GSL / Eigen ) = (" << gsl_spmatrix_nnz(matGSL)  << " / " << matEigen.nonZeros() << ")" << endl;
    for (int k=0; k<matEigen.outerSize(); ++k){
        for (SparseMatrix<double>::InnerIterator it(matEigen, k); it; ++it)
        {
            /*cout << "(" << it.row() << "," << it.col() << ") = "<< it.value() << ",    "
            << gsl_spmatrix_get (C, it.row(), it.col()) << endl;*/
            if(it.value() != gsl_spmatrix_get (C, it.row(), it.col()))
            {
                cout << "Values are not matching: " << gsl_spmatrix_get (C, it.row(), it.col())
                        << " != " << it.value() << endl;
            }
        }
    }


    // Get cols from GSL mat
    /*for (int col = 0; col < mat_size; col++){
        size_t begin_ = C->p[col];
        const size_t end_ = C->p[col+1];
        vector<double> col_vals(end_ - begin_);
        vector<int> row_inds(end_ - begin_);
        size_t curr = 0;
        cout << "For column " << col << ", n neighbors: " << (int)(end_ - begin_) << endl;
        while(begin_ < end_){
            col_vals[curr] = C->data[begin_];
            row_inds[curr++] = C->i[begin_++];
        }
        assert(curr == row_inds.size());

        for(size_t it = 0; it < col_vals.size() ; it++ ){
            cout << "	row: " << row_inds[it] << ", value = " << col_vals[it] << endl;
        }
    }*/

    gsl_spmatrix_free(matGSL);
    gsl_spmatrix_free(C);
    return 0;
}
