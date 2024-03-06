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
#include "calculate_K.hpp"
#include "get_screening_data.hpp"
#include "constants.hpp"
#include "constants.hpp"
#include "run_metadata.hpp"
#include "scf_data.hpp"

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

void get_lower_triangle_three_center_integrals(run_metadata* metadata, scf_data* scfdata,  
 std::vector<double>* three_center_integrals,
 std::vector<int>* sparse_pq_index_map, int unscreened_pq_count){

    int p = metadata->p;
    int Q = metadata->Q;
    int occ = metadata->occ;
    int triangle_length = metadata->triangle_length;
    
    std::vector<int>* triangle_indices = metadata->screened_triangular_indicies;
    
    std::cout << "p = " << p << std::endl;
    std::cout << "Q = " << Q << std::endl;
    std::cout << "occ = " << occ << std::endl;
    std::cout << "triangle_length = " << triangle_length << std::endl;

    //print the sparse_pq_index_map
    std::cout << "sparse_pq_index_map" << std::endl;
    for (int i = 0; i < p; i++){
        for (int j = 0; j < p; j++){
            std::cout << sparse_pq_index_map->data()[get_1d_index(i, j, p)] << " ";
        }
        std::cout << std::endl;
    }
    
    scfdata->three_center_integrals = new std::vector<double>(Q*triangle_length);
    for (int QQ = 0; QQ < Q; QQ++){
        for (int pp = 0; pp < p; pp++){
            for (int qq = 0; qq <= pp; qq++){
               
                int oned_pq_index = get_1d_index(pp, qq, p); //(pq) index
                if (sparse_pq_index_map->data()[oned_pq_index] == 0) //indexes are 1 based from julia
                {
                    continue;
                }
                
                int three_eri_index = get_1d_index(QQ, sparse_pq_index_map->data()[oned_pq_index]-1, unscreened_pq_count); //(Q|pq screened)
                
                int triangle_index = triangle_indices->data()[oned_pq_index];
                int one_d_three_eri_triangle_index = get_1d_index(QQ, triangle_index, triangle_length); //(Q|pq screened triangle)

                // if (pp == 1){
                //     std::cout << " qq = " << qq << " oned_pq_index = " << oned_pq_index <<
                //      " sparse_pq_index_map = " << sparse_pq_index_map->data()[oned_pq_index] << 
                //      " three_eri_index = " << three_eri_index << " triangle_index = " << triangle_index <<
                //      " one_d_three_eri_triangle_index = " << one_d_three_eri_triangle_index << std::endl;
                // }

                scfdata->three_center_integrals->data()[one_d_three_eri_triangle_index] = three_center_integrals->data()[three_eri_index];
            }
        }
    }

    //print the top 10x10 elements of three_center_integrals
    std::cout << "Three center integrals" << std::endl;
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
            std::cout << scfdata->three_center_integrals->data()[get_1d_index(i, j, triangle_length)] << " ";
        }
        std::cout << std::endl;
    }
}



hdf5_data_with_dims<int>* get_sparse_pq_index_map(){
     // read two_center_integrals from file
    return read_file_data<int>(data_path+"sparse_pq_index_map.h5");
}

hdf5_data_with_dims<bool>* get_basis_function_screen_matrix(){

    std::cout << "reading basis function screen matrix" << std::endl;

    // read basis_function_screen_matrix from file
    hdf5_data_with_dims<int>* basis_function_screen_matrix = read_file_data<int>(data_path+"basis_function_screen_matrix.h5");
    hdf5_data_with_dims<bool>* data_bool = new hdf5_data_with_dims<bool>;
    std::cout << "basis_function_screen_matrix->M = " << basis_function_screen_matrix->M << std::endl;
    data_bool->data = new std::vector<bool>(basis_function_screen_matrix->M*basis_function_screen_matrix->N);
    std::cout << "resized data" << std::endl;
    data_bool->M = basis_function_screen_matrix->M;
    data_bool->N = basis_function_screen_matrix->N;

    std::cout << "basis_function_screen_matrix->M = " << basis_function_screen_matrix->M << std::endl;


    std::vector<bool> &bools = *(data_bool->data);
    for (int i = 0; i < basis_function_screen_matrix->M*basis_function_screen_matrix->N; i++){
        bools[i] = basis_function_screen_matrix->data->data()[i] != 0;       
    }
    std::cout << "basis function screen matrix read" << std::endl;
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
    std:: cout << "got basis function screen matrix" << std::endl;
    hdf5_data_with_dims<int>* sparse_pq_index_map = get_sparse_pq_index_map();
    //integral / wavefunction data
    hdf5_data_with_dims<double>* occupied_orbital_coefficients_data = get_occupied_orbital_coefficients();
    hdf5_data_with_dims<double>* two_center_integrals_data = get_two_center_integrals();
    hdf5_data_with_dims<double>* three_center_integrals_data = get_three_center_integrals();

    std::cout << "got data from files" << std::endl;

    run_metadata* metadata = new run_metadata;
    metadata->p = occupied_orbital_coefficients_data->M;
    metadata->Q = two_center_integrals_data->N;
    metadata->occ = occupied_orbital_coefficients_data->N;

    scf_data* scfdata = new scf_data;
    scfdata->two_center_integrals = new std::vector<double>(metadata->Q*metadata->Q);
    //copy from two_center_integrals_data to scfdata
    std::copy(two_center_integrals_data->data->begin(), two_center_integrals_data->data->end(), scfdata->two_center_integrals->begin());

   
    metadata->triangle_length = 0;

    print_start_message(metadata->p, metadata->Q, metadata->occ); 

    //calculate the triangular indicies
    std::cout << "calculating screening metadata" << std::endl;

    get_screening_data(metadata, scfdata, basis_function_screen_matrix_data->data);

    std::cout << "tlength = " << metadata->triangle_length << std::endl;
    std::cout << "triangular indicies calculated" << std::endl;

    //sparse lower trinagle three center integrals (P|pq) where pq is screened and only the lower triangle is stored
  
    //copy occupied orbital coefficients to scfdata
    scfdata->occupied_orbital_coefficients = new std::vector<double>(metadata->p*metadata->occ);
    std::copy(occupied_orbital_coefficients_data->data->begin(), occupied_orbital_coefficients_data->data->end(), scfdata->occupied_orbital_coefficients->begin());


    scfdata->density = new std::vector<double>(metadata->triangle_length);
    scfdata->three_center_integrals = new std::vector<double>(metadata->Q*metadata->triangle_length);
    
    get_lower_triangle_three_center_integrals(metadata, scfdata, three_center_integrals_data->data,
     sparse_pq_index_map->data, three_center_integrals_data->N);
       
    calculate_B(metadata, scfdata, two_center_integrals_data->data); //B is a reference to the updated three_center_integrals_data->data

    std::cout << "calculating density" << std::endl;
    calculate_density(metadata, scfdata,
        basis_function_screen_matrix_data->data);


    std::cout << "print density"    << std::endl;

    std::vector<double>& density = *scfdata->density;
    for(int i =0; i<metadata->triangle_length; i++){
        std::cout <<density[i] << " " << std::endl;
    }

    // //print out the first three center integral 
    // std::cout << "B[0] = " << scfdata->three_center_integrals->data()[0] << std::endl;
    // std::cout << "B[1] = " << scfdata->three_center_integrals->data()[1] << std::endl;
    // std::cout << "twocenterintegrals[1] = " << scfdata->two_center_integrals->data()[1] << std::endl;
    // std::cout << "density[0] = " << scfdata->density->data()[0] << std::endl;

    // // std::cout << "Q: " << Q<< std::endl;
    calculate_J(scfdata, metadata);
    
    // // for (int ii = 0; ii < 10; ii++){
    // //     std::cout << screened_triangular_indices[ii] << " " << std::endl;
    // // }

    calculate_K(scfdata, metadata, basis_function_screen_matrix_data->data);

    delete two_center_integrals_data;
    delete three_center_integrals_data;
    delete basis_function_screen_matrix_data;
    delete sparse_pq_index_map;
    delete occupied_orbital_coefficients_data;
    delete metadata;
    delete scfdata;
    

}


int main() {

    build_fock_cpu();   
    return 0;
}


