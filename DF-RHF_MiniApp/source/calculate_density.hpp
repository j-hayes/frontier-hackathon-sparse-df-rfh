#ifndef __CALCULATE_DENSITY_HPP__
#define __CALCULATE_DENSITY_HPP__

#include "oneapi/mkl/blas.hpp"
#include "mkl.h"
#include "read_hdf5_file.hpp"
#include <string>
#include "index_functions.hpp"
#include "error_checking.hpp"
#include "constants.hpp"

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

double* calculate_density(int p, int occ, double * occupied_orbital_coefficients,
    bool* basis_function_screen_matrix, 
    int* screened_triangular_indices, int screened_triangular_indices_count){
    
    // set cblas_layout to CblasRowMajor
    CBLAS_LAYOUT layout = CblasRowMajor;
    //set transpose to no transpose
    CBLAS_TRANSPOSE transp = CblasNoTrans;


    //allocate density array 
    double* density = new double[p * p];

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

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, p, p, occ, 1.0, 
    occupied_orbital_coefficients, occ, occupied_orbital_coefficients, occ, 0.0, density, p);

    //print the top 10x10 elements of density
    std::cout << "Density" << std::endl;
    
    // for (int i = 0; i < 10; i++){
    //     for (int j = 0; j < 10; j++){
    //         std::cout << density[get_1d_index(i, j, p)] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    int oned_index  = 0;
    double* screened_density = new double[screened_triangular_indices_count];

    std::cout << "Calculating screened density" << std::endl;
    std::cout << "density[0] = " << density[0] << std::endl;
    std::cout << "screened_density[0] = " << screened_density[0] << std::endl;
    std::cout << "basis_function_screen_matrix[0] = " << basis_function_screen_matrix[0] << std::endl;
    std::cout << "screened_triangular_indices[0] = " << screened_triangular_indices[0] << std::endl;

    for (int ii = 0; ii < p; ii++){
        for (int jj = 0; jj < p; jj++){
            if (ii > jj){
                continue;
            }
            oned_index = get_1d_index(jj, ii, p);
            if (basis_function_screen_matrix[oned_index]){
                if(ii == jj){
                    screened_density[screened_triangular_indices[oned_index]] = density[oned_index];
                }
                else{
                    screened_density[screened_triangular_indices[oned_index]] = 2.0*density[oned_index];
                }

            }
        }
    }

    std::cout << "Screened density calculated" << std::endl;


    return screened_density;
}

#endif //__CALCULATE_DENSITY_HPP__