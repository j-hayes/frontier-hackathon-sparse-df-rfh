// simple hello world program

#include <iostream>

#include "cblas.h"
#include "error_checking.hpp"
#include "read_hdf5_file.hpp"

int main() {
    double * A = new double[4];
    A[0] = 1.0;
    A[1] = 2.0;
    A[2] = 3.0;
    A[3] = 4.0;

    double * B = new double[4];
    B[0] = 2.0;
    B[1] = 0.0;
    B[2] = 2.0;
    B[3] = 0.0;

    hdf5_data_with_dims<double>* occupied_orbital_coefficients = 
        read_file_data<double>("/global/u2/j/jhayes1/source/frontier-hackathon-sparse-df-rfh/water_data/occupied_orbital_coefficients.h5");

    std::cout << "occupied_orbital_coefficients[0]" << occupied_orbital_coefficients->data->data()[0] << std::endl;

    double* C = new double[4];

    //cblas_dgemm A and B
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);
    std::cout << "Hello, DF-RHF!" << test_dummy(1,2) << std::endl;
    std::cout << "Hello, DF-RHF!" << C[0] << std::endl;
    return 0;
}