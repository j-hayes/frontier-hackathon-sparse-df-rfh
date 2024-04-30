#ifndef __get_screening_data_hpp
#define __get_screening_data_hpp

#include "index_functions.hpp"
#include "constants.hpp"
#include "run_metadata.hpp"
#include "scf_data.hpp"
#include <vector>
#include <numeric>
#include <iostream>

void get_screening_data(run_metadata* metadata, scf_data* scfdata, std::vector<bool>* basis_function_screen_matrix){

    //initialize the metadata and scfdata vectors
    metadata->triangle_length = 0;
    metadata->non_screened_pq_indices_count = new std::vector<int>(metadata->p, 0);
    metadata->sparse_p_start_indices = new std::vector<int>(metadata->p, 0);
    metadata->screened_triangular_indicies = new std::vector<int>(metadata->p * metadata->p, -1);
    scfdata->non_zero_coefficients = new std::vector<std::vector<double>>();

    std::vector<bool>& basis_function_screen_data = *basis_function_screen_matrix;
    std::vector<int>& screened_triangular_indices = *(metadata->screened_triangular_indicies);
    
    int p_start = 0;
    int oned_index = 0;
    int inverse_index = 0;
    for (int pp = 0; pp < metadata->p; pp++){
        // non_screened_pq_indices_count->data()[pp] = sum of basis_function_screen_matrix[pp, :]
        auto row_start = basis_function_screen_data.begin() + pp * metadata->p;
        auto row_end = row_start + metadata->p;
        // Use std::accumulate to sum the row
        metadata->non_screened_pq_indices_count->data()[pp] = std::accumulate(row_start, row_end, 0);
        metadata->sparse_p_start_indices->data()[pp] = p_start;
        p_start = p_start + metadata->non_screened_pq_indices_count->data()[pp];
        scfdata->non_zero_coefficients->push_back(std::vector<double>(metadata->non_screened_pq_indices_count->data()[pp] * metadata->occ, 0.0));

        for (int qq = 0; qq < metadata->p; qq++){
            oned_index = get_1d_index(pp, qq, metadata->p);
            if (qq >= pp && basis_function_screen_data[oned_index]){
                inverse_index = get_1d_index(qq,pp, metadata->p);
                screened_triangular_indices.data()[oned_index] = metadata->triangle_length;      
                screened_triangular_indices.data()[inverse_index] = metadata->triangle_length;
                (metadata->triangle_length)++;
            }
        }
    }
}



#endif // !__get_screening_data_hpp