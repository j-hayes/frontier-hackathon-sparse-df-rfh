#ifndef __calculate_F_HPP__
#define __calculate_F_HPP__

#include <vector>

#include "index_functions.hpp"

#include "run_metadata.hpp"
#include "scf_data.hpp"

void calculate_F(scf_data* scfdata, run_metadata* metadata, std::vector<bool>* basis_function_screen_matrix){
    int p = metadata->p;
    scfdata->two_electron_fock = new std::vector<double>(p*p);
    std::vector<int>& screened_triangular_indices = *metadata->screened_triangular_indicies;
    std::vector<bool>& basis_function_screen_data = *basis_function_screen_matrix;

    for (int pp = 0; pp < p; pp++){
        for (int qq = 0; qq < p; qq++){
            int one_d_pq_index = get_1d_index(pp, qq, p);
            int inverse_one_d_pq_index = get_1d_index(qq, pp, p);
            
            if (basis_function_screen_data[one_d_pq_index] == false){
                continue;
            }
            int triangular_index = screened_triangular_indices[one_d_pq_index];
            scfdata->two_electron_fock->data()[one_d_pq_index] = 
                scfdata->J->data()[triangular_index] + 
                scfdata->K->data()[one_d_pq_index];

            scfdata->two_electron_fock->data()[inverse_one_d_pq_index] = scfdata->two_electron_fock->data()[one_d_pq_index];
        }
    }

    //print the two electron fock matrix
    std::cout << "Two electron fock matrix" << std::endl;
    for (int pp = 0; pp < 10; pp++){
        for (int qq = 0; qq < 10; qq++){
            std::cout << scfdata->two_electron_fock->data()[get_1d_index(pp, qq, p)] << " ";
        }
        std::cout << std::endl;
    }
}

#endif //__calculate_F_HPP__
