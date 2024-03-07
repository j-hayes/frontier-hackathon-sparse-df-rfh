#ifndef __calculate_J_HPP__
#define __calculate_J_HPP__

//#include "oneapi/mkl/blas.hpp"
//#include "mkl.h"
#include "cblas.h"

#include "index_functions.hpp"
#include "error_checking.hpp"
#include "constants.hpp"
#include "run_metadata.hpp"
#include "scf_data.hpp"

// bool check_V_result(double* result, int Q){
//     //get V data from file 
//     hdf5_data_with_dims<double>* V_data = 
//         read_file_data<double>(data_path+"coulomb_intermediate.h5");

//     std::cout << "Checking V result" << std::endl;
//     std:: cout << "Q = " << Q << std::endl;

//     for (int i = 0; i < Q; i++){        
//         if (values_are_not_same_and_relevant(result[i], V_data->data[i])){
//             std::cout << "V result does not match expected value" << std::endl;
//             std::cout << "calculated_value = " << result[i] << std::endl;
//             std::cout << "expected_value = " << V_data->data[i] << std::endl;
//             std::cout << "index = " << i << std::endl;
//             return false;         
//         }
//     }
//     // std::cout << "V result matches expected result" << std::endl;
//     return true;
// }

// bool check_J_result(double* result, int size){
//     //get J data from file 
//     hdf5_data_with_dims<double>* J_data = 
//         read_file_data<double>(data_path+"J.h5");
//     for (int i = 0; i < size; i++){        
//         if (values_are_not_same_and_relevant(result[i], J_data->data[i])){
//             std::cout << "J result does not match expected value" << std::endl;
//             std::cout << "calculated_value = " << result[i] << std::endl;
//             std::cout << "expected_value = " << J_data->data[i] << std::endl;
//             std::cout << "index = " << i << std::endl;
//             return false;         
//         }
//     }
//     std::cout << "J result matches expected result" << std::endl;
//     return true;
// }


// calculate the J matrix
// V[Q] = B(pq|Q)*density(p|q) coulomb intermediate
// J(pq) = B(pq|Q)*V[Q] coulomb matrix 
// J and density are in 1d triangular stroage with screened out pq ommited

void calculate_J(scf_data* scfdata, run_metadata* metadata){
    
    std::cout << "Calculating J" << std::endl;
    std::cout << "occupied_orbital_coefficients[0] = " << scfdata->occupied_orbital_coefficients->data()[0] << std::endl;
    std::cout << "density[0] = " << scfdata->density->data()[0] << std::endl;
    std::cout << "B[0] = " << scfdata->three_center_integrals->data()[0] << std::endl;
    // for (int i = 0; i < 10; i++){
    //     for (int j = 0; j < 10; j++){
    //         std::cout << scfdata->three_center_integrals[get_1d_index(i, j,  metadata->triangle_length)] << " ";
    //     }
    //     std::cout << std::endl;
    // }



    int p = metadata->p;
    int Q = metadata->Q;
    int triangle_length = metadata->triangle_length;
    
 
    
    // set cblas_layout to CblasRowMajor
    CBLAS_LAYOUT layout = CblasRowMajor;
    //set transpose to no transpose
    CBLAS_TRANSPOSE transp = CblasNoTrans;

    //allocate V array
    // V[Q] = B(Q|pq)*density(p|q) coulomb intermediate
    scfdata->V = new std::vector<double>(Q);
    std::cout << "density[0] = " << scfdata->density->data()[0] << std::endl;
    std::cout << "B[0] = " << scfdata->three_center_integrals->data()[0] << std::endl;
    std::cout << "Q = " << Q << std::endl;
    std::cout << "triangle_length = " << triangle_length << std::endl;
    

    std::cout << "Calculating V" << std::endl;
    // print out the density array 
    for (int i = 0; i < triangle_length; i++){
        std::cout << "density[" << i << "] = " << scfdata->density->data()[i] << std::endl;
    }

    //print the top 10x10 of three_center_integrals
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
            std::cout << scfdata->three_center_integrals->data()[get_1d_index(i, j, triangle_length)] << " ";
        }
        std::cout << std::endl;
    }

    cblas_dgemv(layout, transp, Q, triangle_length, 1.0, scfdata->three_center_integrals->data(),
         triangle_length, scfdata->density->data(), 1, 0.0, scfdata->V->data(), 1);
    
    // std::cout << "Checking V result" << std::endl;    
    // // check_V_result(V, Q);
    // // print the first 20 elements of V
    for (int i = 0; i < Q; i++){
        std::cout << "V[" << i << "] = " << scfdata->V->data()[i] << std::endl;
    }
    scfdata->J = new std::vector<double>(triangle_length);
    
    // J(pq) = B(pq|Q)*V[Q] coulomb matrix 
    cblas_dgemv(layout, CblasTrans, Q, triangle_length, 2.0, scfdata->three_center_integrals->data(),
         triangle_length, scfdata->V->data(), 1, 0.0, scfdata->J->data(), 1);
    
    // std::cout << "Checking J result" << std::endl;

    // // check_J_result(J, triangle_length);
    // //print the first 20 elements of J
    for (int i = 0; i < triangle_length; i++){
        std::cout << "J[" << i << "] = " << scfdata->J->data()[i] << std::endl;
    }
}

#endif //__calculate_J_HPP__