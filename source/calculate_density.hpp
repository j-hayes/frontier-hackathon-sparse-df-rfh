#ifndef __CALCULATE_DENSITY_HPP__
#define __CALCULATE_DENSITY_HPP__

//#include "oneapi/mkl/blas.hpp"
//#include "mkl.h"
#include "cblas.h"

#include "read_hdf5_file.hpp"
#include <string>
#include "index_functions.hpp"
#include "error_checking.hpp"
#include "constants.hpp"
#include "run_metadata.hpp"
#include "scf_data.hpp"

bool check_density_result(double* result, int size, double tol){
    //get density data from file 
    hdf5_data_with_dims<double>* density_data = 
        read_file_data<double>(data_path+"density.h5");
    for (int i = 0; i < size; i++){
        for (int j = 0; j < i; j++){

            // if(values_are_not_same_and_relevant(result[get_1d_index(i, j, size)], density_data->data[get_1d_index(i, j, size)])){
            //     std::cout << "Density result does not match expected value" << std::endl;
            //     std::cout << "calculated_value = " << result[get_1d_index(i, j, size)] << std::endl;
            //     std::cout << "expected_value = " << density_data->data[get_1d_index(i, j, size)] << std::endl;
            //     std::cout << "index = " << get_1d_index(i, j, size) << std::endl;
            //     return false;
            // }
        }
    }
    return true;
}

void calculate_density(run_metadata* metadata, scf_data* scfdata,
    std::vector<bool>* basis_function_screen_matrix){
    
    // set cblas_layout to CblasRowMajor
    CBLAS_LAYOUT layout = CblasRowMajor;
    //set transpose to no transpose
    CBLAS_TRANSPOSE transp = CblasNoTrans;
    std::vector<int>* screened_triangular_indices = metadata->screened_triangular_indicies;
    int p = metadata->p;
    int occ = metadata->occ;
    int screened_triangular_indices_count = metadata->triangle_length;



    //allocate density array 
    std::vector<double>* density = new std::vector<double>(p*p);

    // std::cout << "Calculating density" << std::endl;
    std::cout << "p = " << p << std::endl;
    std::cout << "occ = " << occ << std::endl;
    std::cout << "screened_triangular_indices_count = " << screened_triangular_indices_count << std::endl;

    //print the occupied_orbital_coefficients 
    // for (int i = 0; i < p; i++){
    //     for (int j = 0; j < occ; j++){
    //         std::cout << occupied_orbital_coefficients[get_1d_index(i, j, p)] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    int M = p;
    int N = p;
    int K = occ;
    int lda = occ;
    int ldb = occ;
    int ldc = p;
    double alpha = 1.0;
    double beta = 0.0;

    //cblas_dgemm A and B
    cblas_dgemm(CblasRowMajor , CblasNoTrans, CblasTrans, M, N, K, 
    alpha, scfdata->occupied_orbital_coefficients->data(), lda, 
    scfdata->occupied_orbital_coefficients->data(), ldb, 
    beta, density->data(), ldc);

    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, p, p, occ, 
    // 1.0,  scfdata->occupied_orbital_coefficients->data(), occ, 
    // scfdata->occupied_orbital_coefficients->data(), occ, 
    // 0.0, density->data(), p);

    //print the top 10x10 elements of density
    std::cout << "Density" << std::endl;
    
    // for (int i = 0; i < 10; i++){
    //     for (int j = 0; j < 10; j++){
    //         std::cout << density[get_1d_index(i, j, p)] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    int oned_index  = 0;
    std::vector<bool>& screen_vector = *basis_function_screen_matrix;
    std::vector<int>& triangular_indicies_vector = *screened_triangular_indices;
    std::vector<double>& density_vector = *(scfdata->density);
    for (int ii = 0; ii < p; ii++){
        for (int jj = 0; jj < p; jj++){
            if (ii > jj){
                continue;
            }
            oned_index = get_1d_index(jj, ii, p);
            if (screen_vector[oned_index]){
                if(ii == jj){
                    density_vector[triangular_indicies_vector[oned_index]] = 1.0*density->data()[oned_index];
                }
                else{
                    density_vector[triangular_indicies_vector[oned_index]] = 2.0*density->data()[oned_index];
                }
             }
        }
    }
    std::cout << "Screened density calculated" << std::endl;
}

#endif //__CALCULATE_DENSITY_HPP__