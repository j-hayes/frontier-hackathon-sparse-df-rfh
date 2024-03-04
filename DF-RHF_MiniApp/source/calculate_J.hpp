#ifndef __calculate_J_HPP__
#define __calculate_J_HPP__

#include "oneapi/mkl/blas.hpp"
#include "mkl.h"

#include "index_functions.hpp"
#include "error_checking.hpp"
#include "constants.hpp"

bool check_V_result(double* result, int Q){
    //get V data from file 
    hdf5_data_with_dims<double>* V_data = 
        read_file_data<double>(data_path+"coulomb_intermediate.h5");

    std::cout << "Checking V result" << std::endl;
    std:: cout << "Q = " << Q << std::endl;

    for (int i = 0; i < Q; i++){        
        if (values_are_not_same_and_relevant(result[i], V_data->data[i])){
            std::cout << "V result does not match expected value" << std::endl;
            std::cout << "calculated_value = " << result[i] << std::endl;
            std::cout << "expected_value = " << V_data->data[i] << std::endl;
            std::cout << "index = " << i << std::endl;
            return false;         
        }
    }
    // std::cout << "V result matches expected result" << std::endl;
    return true;
}

bool check_J_result(double* result, int size){
    //get J data from file 
    hdf5_data_with_dims<double>* J_data = 
        read_file_data<double>(data_path+"J.h5");
    for (int i = 0; i < size; i++){        
        if (values_are_not_same_and_relevant(result[i], J_data->data[i])){
            std::cout << "J result does not match expected value" << std::endl;
            std::cout << "calculated_value = " << result[i] << std::endl;
            std::cout << "expected_value = " << J_data->data[i] << std::endl;
            std::cout << "index = " << i << std::endl;
            return false;         
        }
    }
    std::cout << "J result matches expected result" << std::endl;
    return true;
}


// calculate the J matrix
// V[Q] = B(pq|Q)*density(p|q) coulomb intermediate
// J(pq) = B(pq|Q)*V[Q] coulomb matrix 
// J and density are in 1d triangular stroage with screened out pq ommited

double* calculate_J(double * B, double * density, int Q, int triangle_length){
    
    // set cblas_layout to CblasRowMajor
    CBLAS_LAYOUT layout = CblasRowMajor;
    //set transpose to no transpose
    CBLAS_TRANSPOSE transp = CblasNoTrans;

    //allocate V array
    double* V = new double[Q];
    // V[Q] = B(Q|pq)*density(p|q) coulomb intermediate

    // print size of J
    cblas_dgemv(layout, transp, Q, triangle_length, 1.0, B, triangle_length, density, 1, 0.0, V, 1);
    
    double* J = new double[triangle_length];
    
    // // J(pq) = B(pq|Q)*V[Q] coulomb matrix 
    cblas_dgemv(layout, CblasTrans, Q, triangle_length, 2.0, B, triangle_length, V, 1, 0.0, J, 1);
    
    
    // check_J_result(J, triangle_length);
    //print the first 20 elements of J

    return J;

}

#endif //__calculate_J_HPP__