#ifndef calculate_triangular_indicies_hpp
#define calculate_triangular_indicies_hpp

#include "index_functions.hpp"
#include "constants.hpp"
#include "run_metadata.hpp"
#include <vector>

std::vector<int>* calculate_triangular_indicies(run_metadata* metadata, std::vector<bool>* basis_function_screen_matrix){
    
    std::vector<int>* screened_triangular_indices = new std::vector<int>(metadata->p * metadata->p); 
    int oned_index = 0;
    int inverse_index = 0;

    // std::cout << "basis_function_screen_matrix[0] = " << basis_function_screen_matrix[0] << std::endl;
    // std::cout << "basis_function_screen_matrix[1] = " << basis_function_screen_matrix[1] << std::endl;
    // std::cout << "p = " << p << std::endl;

    std::vector<bool>& basis_function_screen_data = *basis_function_screen_matrix;
    for (int ii = 0; ii < metadata->p; ii++){
        for (int jj = 0; jj < metadata->p; jj++){
            oned_index = get_1d_index(ii, jj, metadata->p);
            if (jj >= ii && basis_function_screen_data[oned_index]){
                inverse_index = get_1d_index(jj,ii, metadata->p);
                screened_triangular_indices->data()[oned_index] = metadata->triangle_length;      
                screened_triangular_indices->data()[inverse_index] = metadata->triangle_length;
                (metadata->triangle_length)++;
            }
        }
    }

    
    return screened_triangular_indices;
}

#endif // !calculate_triangular_indicies_hpp

