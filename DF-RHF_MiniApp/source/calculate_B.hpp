#ifndef __calcualte_B_hpp
#define __calcualte_B_hpp

#include "oneapi/mkl/blas.hpp"
#include "mkl.h"
#include "oneapi/mkl/lapack.hpp"

#include "index_functions.hpp"
#include "read_hdf5_file.hpp"
#include <string>
#include "error_checking.hpp"
#include "constants.hpp"

//check J_AB_inv result
bool check_J_AB_inv_result(double* two_center_integrals, int Q){
    //get J_AB_inv data from file 
    hdf5_data_with_dims<double>* J_AB_inv_data = 
        read_file_data<double>(data_path+"J_AB_INV.h5");

    // copy the lower triangle to the upper triangle for simplicity 
    for (int i = 0; i < Q; i++){
        for (int j = 0; j < Q; j++){
            J_AB_inv_data->data[get_1d_index(i, j, Q)] = J_AB_inv_data->data[get_1d_index(j, i, Q)];
        }
    }
    double calculated_value;
    double expected_value;
    for (int i = 0; i < Q; i++){
        for (int j = 0; j < Q; j++){
            if (i > j){
                continue;
            }
            calculated_value = two_center_integrals[get_1d_index(i, j, Q)];
            expected_value = J_AB_inv_data->data[get_1d_index(i, j, Q)];
            //check to see if the values are approximately equal
            if (values_are_not_same_and_relevant(calculated_value, expected_value)){
                std::cout << "J_AB_inv result does not match expected value" << std::endl;
                std::cout << "calculated_value = " << calculated_value << std::endl;
                std::cout << "expected_value = " << expected_value << std::endl;
                return false;
            }
        
        }
    }
    return true;


}

bool check_B_result(double* three_center_integrals, int Q, int triangle_length){
    //get B data from file 

    //todo need triangle index map for checking

    hdf5_data_with_dims<double>* B_data = 
        read_file_data<double>(data_path+"B_Triangle.h5");


    //print B_data M and N
    int incorrects = 0;
    double calculated_value;
    double expected_value;
    for (int i = 0; i < Q; i++){
        for (int j = 0; j < triangle_length; j++){
            calculated_value = three_center_integrals[get_1d_index(i, j, triangle_length)];
            expected_value = B_data->data[get_1d_index(i, j, triangle_length)]; 
            //check to see if the values are approximately equal
            if (values_are_not_same_and_relevant(calculated_value, expected_value)){
                std::cout << "B result does not match expected value" << std::endl;
                std:: cout << "i = " << i << " j = " << j << std::endl;
                std::cout << "calculated_value = " << calculated_value << std::endl;
                std::cout << "expected_value =  " << expected_value << std::endl;
                incorrects +=1;
                if (incorrects > 100){
                    std::cout << "Too many incorrects" << std::endl;
                    return false;
                }
            }
        }
    }
    std::cout << "B result matches expected result" << std::endl;
    return true;
}


//two_center_integrals is a 1d array of size (P|Q)
//three_center_integrals is a 1d array of size (P|pq) with the primary indicies being in lower triangular screened storage
void calculate_B(int p, int Q, double* two_center_integrals, double* three_center_integrals, int triangle_length){
    
    int info;
    
    // !!!!! CHOLESKY DECOMPOSION OF 2C-2E MATRIX V=L*LT
    dpotrf("L", &Q, two_center_integrals, &Q, &info);
    std::cout << "Done with cholesky decomp" << std::endl;
    // !!!!! DETERMINATION OF INVERSE OF CHOLESKY DECOMPOSED MATRIX L^(-1)
    dtrtri("L", "N", &Q, two_center_integrals, &Q, &info);
    std::cout << "Done with inverse of cholesky decomp" << std::endl;

    // bool J_AB_inv_is_correct = check_J_AB_inv_result(two_center_integrals, Q);

    //calculate B 
    // B(P|pq) = (P|Q)*(Q|pq) (reuse the three_center_integrals array for B)
    // set cblas_layout to CblasRowMajor
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, Q, triangle_length, 1.0, two_center_integrals,
        Q, three_center_integrals, triangle_length);
}


#endif // !__calcualte_B_hpp


