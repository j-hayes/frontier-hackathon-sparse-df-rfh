#ifndef __calculate_K_HPP__ 
#define __calculate_K_HPP__

#include "oneapi/mkl/blas.hpp"
#include "mkl.h"

#include "index_functions.hpp"
#include "error_checking.hpp"
#include "constants.hpp"
#include "run_metadata.hpp"
#include "scf_data.hpp"

void calculate_K(scf_data* scfdata, run_metadata* metadata, std::vector<bool>* basis_function_screen_matrix){
    std::cout << "Calculating K" << std::endl;
    
    int p = metadata->p;
    int occ = metadata->occ;
    int Q = metadata->Q;
    int triangle_length = metadata->triangle_length;
    
    //allocate W 
    scfdata->W = new std::vector<double>(p*occ*Q);
    std::vector<std::vector<double>>& non_zero_coefficients = *scfdata->non_zero_coefficients;
    std::vector<bool>& basis_function_screen_data = *basis_function_screen_matrix;
    std::vector<int>& screened_triangular_indices = *(metadata->screened_triangular_indicies);
    std::vector<int>& non_screened_pq_indices_count = *(metadata->non_screened_pq_indices_count);
    std::vector<double> B_p = std::vector<double>(Q*p); //buffer of Q*p is the maximum size of B_p could be could also use max(metadata->non_screened_pq_indices_count)
    for(int pp=0; pp<p; pp++){
        int non_zero_r_index = 0;
        for(int r=0; r<p; r++){
            int one_d_pr_index = get_1d_index(r, pp, p);
            if(basis_function_screen_data[one_d_pr_index] == false){
                continue;
            }
            for(int i=0; i<occ; i++){
                non_zero_coefficients.data()[pp][get_1d_index(non_zero_r_index, i, occ)] 
                    = scfdata->occupied_orbital_coefficients->data()[get_1d_index(r, i, p)];
            }
            non_zero_r_index++;  
            for(int QQ=0; QQ<Q; QQ++) {//brute force probably slow hopefully sparse dgemm can alleviate the need for this?
                int triangular_screened_index = screened_triangular_indices[one_d_pr_index];
                B_p[get_1d_index(QQ, r, non_screened_pq_indices_count[pp])]
                    = scfdata->three_center_integrals->data()[get_1d_index(QQ, triangular_screened_index, triangle_length)]; 
            }
        }           


    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Q, occ, non_screened_pq_indices_count[pp], 
        1.0, B_p.data(), Q,
        non_zero_coefficients.data()[pp].data(), non_screened_pq_indices_count[pp], 0.0, scfdata->W->data()+ pp*Q*occ, Q);
    }


    //print the top 10x10 of W
    std::cout << "W" << std::endl;
    
    for(int QQ=0; QQ<10; QQ++){
        for(int ii=0; ii<occ; ii++){
            int Qi_index = get_1d_index(ii, QQ, Q);
            int pQi_index = get_1d_index(0, Qi_index, Q*occ);
            std::cout << "W[" << 0 << "," << ii << "," << QQ << "] = "
              << scfdata->W->data()[pQi_index] << " " << std::endl;
        }
    }
}

/*

function calculate_W_from_trianglular_screened(B, p,i,occupied_orbital_coefficients, basis_function_screen_matrix, screened_triangular_indices, non_zero_coefficients, W)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    Threads.@threads for pp in 1:p
        non_zero_r_index = 1
        for r in eachindex(view(screened_triangular_indices, :, pp))
            if basis_function_screen_matrix[r,pp] 
                non_zero_coefficients[pp][non_zero_r_index, :] .= occupied_orbital_coefficients[r,:] 
                non_zero_r_index += 1
            end
        end
        indices = [x for x in view(screened_triangular_indices, :, pp) if x != 0]
        BLAS.gemm!('N', 'N', 1.0,  B[:, indices], non_zero_coefficients[pp], 0.0, view(W, :, :, pp))
    end
    BLAS.set_num_threads(blas_threads)
end
*/


#endif //__calculate_K_HPP__ 