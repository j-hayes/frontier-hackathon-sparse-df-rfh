// // stl includes
// #include <iostream>
// #include <cstdlib>
// #include <limits>
// #include <vector>
// #include <algorithm>
// #include <cstring>
// #include <list>
// #include <iterator>

// #include <CL/sycl.hpp>
#include "oneapi/mkl/blas.hpp"
#include "mkl.h"
#include "read_hdf5_file.hpp"
#include <string>
#include "mkl.h"
#include "index_functions.hpp"
#include "calculate_density.hpp"
#include "calculate_B.hpp"
#include "calculate_J.hpp"
#include "calculate_triangular_indicies.hpp"
#include "constants.hpp"
#include "constants.hpp"


void print_start_message(int p, int Q, int occ){
    std::cout << "Starting DF-RHF MiniApp" << std::endl;
    std::cout << "Reading data from file" << std::endl;
    std::cout << "p = " << p << std::endl;
    std::cout << "Q = " << Q << std::endl;
    std::cout << "occ = " << occ << std::endl;
}

hdf5_data_with_dims<double>* get_two_center_integrals(){
     // read two_center_integrals from file
    return read_file_data<double>(data_path+"two_center_integrals.h5");
}


hdf5_data_with_dims<double>* get_three_center_integrals(){
    // read three_center_integrals from file 
    return read_file_data<double>(data_path+"three_center_integrals_Q_by_pq.h5");
}

double* get_lower_triangle_three_center_integrals(double* three_center_integrals, int unscreened_pq_count,
    int* triangle_indices, int Q, int p, int triangle_length, int* sparse_pq_index_map){

    std::cout << "triangle length " << triangle_length << std::endl;
    std::cout << "unscreened_pq_count " << unscreened_pq_count << std::endl;


    double* lower_triangle_three_center_integrals = new double[Q*triangle_length];
    for (int QQ = 0; QQ < Q; QQ++){
        for (int pp = 0; pp < p; pp++){
            for (int qq = 0; qq <= pp; qq++){
               
                int oned_pq_index = get_1d_index(pp, qq, p); //(pq) index
                if (sparse_pq_index_map[oned_pq_index] == 0) //indexes are 1 based from julia
                {
                    continue;
                }
                
                int three_eri_index = get_1d_index(QQ, sparse_pq_index_map[oned_pq_index]-1, unscreened_pq_count); //(Q|pq screened)
                
                int triangle_index = triangle_indices[oned_pq_index];
                int one_d_three_eri_triangle_index = get_1d_index(QQ, triangle_index, triangle_length); //(Q|pq screened triangle)

                lower_triangle_three_center_integrals[one_d_three_eri_triangle_index] = three_center_integrals[three_eri_index];
            }
        }
    }

  
    return lower_triangle_three_center_integrals;
}



hdf5_data_with_dims<int>* get_sparse_pq_index_map(){
     // read two_center_integrals from file
    return read_file_data<int>(data_path+"sparse_pq_index_map.h5");
}

hdf5_data_with_dims<bool>* get_basis_function_screen_matrix(){
    // read basis_function_screen_matrix from file
    hdf5_data_with_dims<int>* data = read_file_data<int>(data_path+"basis_function_screen_matrix.h5");
    hdf5_data_with_dims<bool>* data_bool;
    data_bool = new hdf5_data_with_dims<bool>;
    data_bool->data = new bool[data->M*data->N];
    data_bool->M = data->M;
    data_bool->N = data->N;

    for (int i = 0; i < data->M*data->N; i++){
        data_bool->data[i] = data->data[i] != 0;       
    }

    return data_bool;

}

hdf5_data_with_dims<double>* get_occupied_orbital_coefficients(){
    // read occupied_orbital_coefficients from file
    return read_file_data<double>(data_path+"occupied_orbital_coefficients.h5");
}

hdf5_data_with_dims<double>* get_B_from_file(){
    // read B from file
    return read_file_data<double>(data_path+"B_Triangle.h5");
}

void modify_Q(int Q){
    Q = 10;
}

// reads data from files and calls the various functions to build the fock matrix checks the results against known fock matrix
void build_fock_cpu(){

    //load the test input data from files    
    //screening data
    hdf5_data_with_dims<bool>* basis_function_screen_matrix_data = get_basis_function_screen_matrix();
    hdf5_data_with_dims<int>* sparse_pq_index_map = get_sparse_pq_index_map();

    //integral / wavefunction data
    hdf5_data_with_dims<double>* occupied_orbital_coefficients_data = get_occupied_orbital_coefficients();
    hdf5_data_with_dims<double>* two_center_integrals_data = get_two_center_integrals();
    hdf5_data_with_dims<double>* three_center_integrals_data = get_three_center_integrals();


    int p = occupied_orbital_coefficients_data->M;
    int Q = two_center_integrals_data->N;
    int occ = occupied_orbital_coefficients_data->N;
    int triangle_length = 28;

    int* tlength;
        
    print_start_message(p,Q,occ);   

    //calculate the triangular indicies
    std::cout << "calculating triangular indicies" << std::endl;

    int* screened_triangular_indices = calculate_triangular_indicies(p,
         basis_function_screen_matrix_data->data, tlength);

    // std::cout << "tlength = " << tlength << std::endl;
    std::cout << "triangular indicies calculated" << std::endl;

    //sparse lower trinagle three center integrals (P|pq) where pq is screened and only the lower triangle is stored
    double* lower_triangle_three_center_integrals = get_lower_triangle_three_center_integrals(three_center_integrals_data->data, 
        three_center_integrals_data->N, screened_triangular_indices, Q, p, 
        triangle_length, sparse_pq_index_map->data);
    
    double* density = calculate_density(p, occ, occupied_orbital_coefficients_data->data,
    basis_function_screen_matrix_data->data, 
    screened_triangular_indices, triangle_length);

    //print the density array 
    std::cout << "Density" << std::endl;
    for (int i = 0; i < triangle_length; i++){        
        std::cout << density[i] << " " << std::endl;
    }
    
    
    std::cout << "calcualting B" << std::endl;
    calculate_B(p, Q, two_center_integrals_data->data, lower_triangle_three_center_integrals, triangle_length); //B is a reference to the updated three_center_integrals_data->data
    
   
    std::cout << "B top 10x10" << std::endl;
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
            std::cout << lower_triangle_three_center_integrals[get_1d_index(i, j, triangle_length)] << " ";
        }
        std::cout << std::endl;
    }
    
        

    // std::cout << "Q: " << Q<< std::endl;
    double* J = calculate_J(lower_triangle_three_center_integrals, density, Q, triangle_length);

    // for (int ii = 0; ii < 10; ii++){
    //     std::cout << screened_triangular_indices[ii] << " " << std::endl;
    // }

}


int main() {

    build_fock_cpu();   
    return 0;
}


