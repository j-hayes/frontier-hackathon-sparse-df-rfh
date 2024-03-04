#ifndef calculate_triangular_indicies_hpp
#define calculate_triangular_indicies_hpp

#include "index_functions.hpp"
#include "constants.hpp"

int* calculate_triangular_indicies(int p, bool* basis_function_screen_matrix, int* triangle_length){
    int* screened_triangular_indices = new int[p*p];

    int screened_triangular_indices_count = 0;//todo move this so it is calculated in setup step not calculation step
    int oned_index = 0;
    int inverse_index = 0;

    // std::cout << "basis_function_screen_matrix[0] = " << basis_function_screen_matrix[0] << std::endl;
    // std::cout << "basis_function_screen_matrix[1] = " << basis_function_screen_matrix[1] << std::endl;
    // std::cout << "p = " << p << std::endl;

    for (int ii = 0; ii < p; ii++){
        for (int jj = 0; jj < p; jj++){
            oned_index = get_1d_index(ii, jj, p);
            if (jj >= ii && basis_function_screen_matrix[oned_index]){
                inverse_index = get_1d_index(jj,ii, p);
                // std::cout << "ii = " << ii << " jj = " << jj << " oned_index = " << oned_index
                //      << " inverse_index = " << inverse_index 
                //      << "screened_triangular_indices_count " << screened_triangular_indices_count 
                //      << std::endl;
                screened_triangular_indices[oned_index] = screened_triangular_indices_count;      
                screened_triangular_indices[inverse_index] = screened_triangular_indices_count;
                screened_triangular_indices_count++;
            }
        }
    }

    std::cout << "done with calculating screened_triangular_indices" << std::endl;
    std::cout << "screened_triangular_indices_count = " << screened_triangular_indices_count << std::endl;
    // *triangle_length = screened_triangular_indices_count;
    return screened_triangular_indices;
}

#endif // !calculate_triangular_indicies_hpp

