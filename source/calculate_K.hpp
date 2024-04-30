#ifndef __calculate_K_HPP__ 
#define __calculate_K_HPP__

//#include "oneapi/mkl/blas.hpp"
//#include "mkl.h"
#include "cblas.h"

#include "index_functions.hpp"
#include "error_checking.hpp"
#include "constants.hpp"
#include "run_metadata.hpp"
#include "scf_data.hpp"
#include <iomanip>

struct ScreeningRanges {
    std::vector<std::vector<int>*> r_range_starts;
    std::vector<std::vector<int>*> range_lengths;
    std::vector<std::vector<int>*> B_range_starts;
    std::vector<int>* number_of_ranges;

    void initialize(int p) {
        r_range_starts.resize(p);
        range_lengths.resize(p);
        B_range_starts.resize(p);
        number_of_ranges = new std::vector<int>();
        number_of_ranges->resize(p);
        for (int i = 0; i < p; i++) {
            r_range_starts[i] = new std::vector<int>();
            range_lengths[i] = new std::vector<int>();
            B_range_starts[i] = new std::vector<int>();
            (*number_of_ranges)[i] = 0;        
        }
    }
};

void calculate_ranges(ScreeningRanges ranges, 
    std::vector<std::pair<int, int>> screened_triangular_indicies_to_2d, std::vector<int>& screened_triangular_indicies,
     int p)
{

    std::cout << "starting calculate_ranges pp loop" << std::endl;
    for (int pp = 0; pp < p; pp++) {
        std::vector<int> indices;
        // std::cout << "pp: " << pp << std::endl;
        for (int qq = 0; qq < p; qq++) {
            int oned_index = get_1d_index(pp, qq, p);
            if (screened_triangular_indicies[oned_index] >= 0) {
                indices.push_back(screened_triangular_indicies[oned_index]);
            }
        } // end of qq loop

        //print indicies 
        // std::cout << "indices: ";
        // for (int i = 0; i < indices.size(); i++){
        //     std::cout << indices[i] << ", ";
        // }
        // std::cout << std::endl;

        std::vector<int>& r_ranges_starts = *ranges.r_range_starts[pp];
        std::vector<int>& range_lengths = *ranges.range_lengths[pp];
        std::vector<int>& B_ranges_starts = *ranges.B_range_starts[pp];
        

        int B_start = indices[0];
        int B_last = indices.back();
        std::pair<int, int> r_start = screened_triangular_indicies_to_2d[B_start];
        std::pair<int, int> r_last = screened_triangular_indicies_to_2d[B_last];
        int r_range_start = 0;
        int r_range_end = 0;
        int r_range_next = 0;

        for (int i = 0; i < indices.size() - 1; i++) {
            std::pair<int, int> r_tuple = screened_triangular_indicies_to_2d[indices[i]];
            r_range_end = (r_tuple.first == pp) ? r_tuple.second : r_tuple.first;
            std::pair<int, int> r_tuple_next = screened_triangular_indicies_to_2d[indices[i+1]];
            r_range_next = (r_tuple_next.first == pp) ? r_tuple_next.second : r_tuple_next.first;

            // if (pp == 0) {
            //     std::cout << "r_tuple.first " << r_tuple.first << " r_tuple.second " << r_tuple.second << std::endl;
            //     std::cout << "r_tuple_next.first " << r_tuple_next.first << " r_tuple_next.second " << r_tuple_next.second << std::endl;
            //     std::cout << "indicies[i+1] " << indices[i+1] << " indices[i] " << indices[i] << " r_range_next " << r_range_next << " r_range_end " << r_range_end << std::endl; 
            // }
          
            if (indices[i+1] - indices[i] != 1 || r_range_next - r_range_end != 1) {
                r_range_start = (r_start.first == pp) ? r_start.second : r_start.first;
                

                r_ranges_starts.push_back(r_range_start);
                range_lengths.push_back(r_range_end - r_range_start + 1);
                B_ranges_starts.push_back(B_start);
                // r_ranges.push_back(r_range_end);
                // B_ranges.push_back(B_start);
                // B_ranges.push_back(indices[i]);

                B_start = indices[i+1];
                if (pp == 0) {std::cout << "setting new B_start " << B_start << std::endl;}

                r_start = screened_triangular_indicies_to_2d[indices[i+1]];
                (*ranges.number_of_ranges)[pp] += 1;
                if(pp == 0){
                    // std::cout << "added range r" << r_range_start << " to " << r_range_end << std::endl;
                }
            }
        }
        B_ranges_starts.push_back(B_start);
        // B_ranges.push_back(B_start);
        // B_ranges.push_back(B_last);

        r_range_start = (r_start.first == pp) ? r_start.second : r_start.first;
        r_range_end = (r_last.first == pp) ? r_last.second : r_last.first;
        
        r_ranges_starts.push_back(r_range_start);
        range_lengths.push_back(r_range_end - r_range_start + 1);
        (*ranges.number_of_ranges)[pp] += 1;


        // r_ranges.push_back(r_range_start);
        // r_ranges.push_back(r_range_end);

    } // end of pp loop

    // print the ranges for debugging
    // for (int pp = 0; pp < pp; pp++) {
    // for (int pp = 0; pp < 1; pp++) {
    //     std::cout << "pp: " << pp << std::endl;
    //     for (int i = 0; i < ranges.r_range_starts[pp]->size(); i++) {
    //         auto range_start = (*ranges.r_range_starts[pp])[i];
    //         auto range_length = (*ranges.range_lengths[pp])[i];
    //         auto B_range_start = (*ranges.B_range_starts[pp])[i];
    //         std::cout << "range_start: " << range_start << " range_end: " << range_start + range_length << " B_range_start: " << B_range_start 
    //         << " B_range_end: " << B_range_start + range_length << std::endl;

    //     }
    // }
}


void calculate_K(scf_data* scfdata, run_metadata* metadata, std::vector<bool>* basis_function_screen_matrix){
    std::cout << "Calculating K" << std::endl;
    
    int p = metadata->p;
    int occ = metadata->occ;
    int Q = metadata->Q;
    int triangle_length = metadata->triangle_length;
    
    //allocate W 
    scfdata->W = new std::vector<double>(p*occ*Q);
    //allocate K
    scfdata->K = new std::vector<double>(p*p);
    std::vector<std::vector<double>>& non_zero_coefficients = *scfdata->non_zero_coefficients;
    std::vector<bool>& basis_function_screen_data = *basis_function_screen_matrix;
    std::vector<int>& screened_triangular_indicies = *(metadata->screened_triangular_indicies);
    std::vector<int>& non_screened_pq_indices_count = *(metadata->non_screened_pq_indices_count);

    // precalulate the ranges //TODO: move this to the get_screening_data.hpp
    std::vector<std::pair<int, int>> screened_triangular_indicies_to_2d = std::vector<std::pair<int, int>>();
    screened_triangular_indicies_to_2d.resize(triangle_length);
    std::cout << "Precalculating ranges triangle length: " << triangle_length  << std::endl;
    int triangular_indices_count = 0;
    for (int pp = 0; pp < p; pp++) {
        for (int qq = 0; qq < p; qq++) {
            int oned_index = get_1d_index(pp, qq, p);
            if (pp <= qq && basis_function_screen_data[oned_index]) {
                screened_triangular_indicies_to_2d[triangular_indices_count] = std::make_pair(pp, qq);
                triangular_indices_count++;

            }
        }
    }

    ScreeningRanges ranges;
        std::cout << "Calculating ranges" << std::endl;
    ranges.initialize(p);
    calculate_ranges(ranges, screened_triangular_indicies_to_2d, *(metadata->screened_triangular_indicies), p);

    std::vector<double> occ_T = std::vector<double>(occ*p, 0.0);
    for (int i = 0; i < occ; i++){
        for (int j = 0; j < p; j++){
            occ_T[get_1d_index(i, j, occ)] = scfdata->occupied_orbital_coefficients->data()[get_1d_index(j, i, occ)];
        }
    }

    double alpha = 1.0;
    double beta = 1.0;
    int M = 0;
    int N = 0;
    int K = 0;
    int ldb = 0;
    int ldc = 0;
    int lda = 0; 
    //trans A is transpose 
    auto transA = CblasTrans;
    auto transB = CblasNoTrans;

    

    // auto W_new = new std::vector<double>(p*occ*Q, 0.0);
   
    M = occ;
    N = Q;
    lda = occ;
    ldb = Q;
    ldc = Q;
    for (int pp = 0; pp < p; pp++) {
        // std::cout << "pp: " << pp << "number of ranges" << (*ranges.number_of_ranges)[pp] << std::endl;
        for (int range_index = 0; range_index < (*ranges.number_of_ranges)[pp]; range_index++) {
            //dgemm_blas call of transpose(K by occ) * K by Q = occ by Q
            K = (*ranges.range_lengths[pp])[range_index];
            
            int r_range_start = (*ranges.r_range_starts[pp])[range_index];
            int B_range_start = (*ranges.B_range_starts[pp])[range_index];
            // std::cout << range_index << " r_range_start: " << r_range_start << " B_range_start: " << B_range_start << std::endl;
            cblas_dgemm(CblasRowMajor, transA, transB, M, N, K, alpha, scfdata->occupied_orbital_coefficients->data() + r_range_start*occ, lda,
            scfdata->three_center_integrals_T->data() + Q*B_range_start, ldb, beta,
            scfdata->W->data() + pp*occ*Q, ldc);

      
        }
    }

    //print the first element of W_new
    // std::cout << "first element in W_new = " << W_new->data()[0] << std::endl;

    //calculate K_new 
    M = p;
    N = p;
    K = Q*occ;
    lda = K; // Columns of the matrix A for row-major cblas_dgemm
    ldb = K; // Columns of the matrix B for row-major cblas_dgemm
    ldc = N; // Columns of the matrix C for row-major cblas_dgemm
    alpha = -1.0;
    beta = 0.0;

    //the dimensions of W in 3d are p x occ x Q
    // auto K_new = new std::vector<double>(p*p,0.0);
    // Perform the multiplication
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, scfdata->W->data(), lda, 
        scfdata->W->data(), ldb, 
        beta, scfdata->K->data(), ldc);

    //print the top 10x10 elements of K_new with fixed width of 10 characters
    std::cout << "K_new" << std::endl;
    for (int i = 0; i < 20; i++){
        for (int j = 0; j < 20; j++){
            std::cout << std::setw(10) << scfdata->K->data()[get_1d_index(i, j, p)] << " ";
        }
        std::cout << std::endl;
    }
    
  
    // std::vector<double> B_p = std::vector<double>(Q*p, 0.0); //buffer of Q*p is the maximum size of B_p could be could also use max(metadata->non_screened_pq_indices_count)
    
    // alpha = 1.0;
    // beta = 0.0;
    
    // for(int pp=0; pp<p; pp++){
    //     int non_zero_r_index = 0;
    //     for(int r=0; r<p; r++){
    //         int one_d_pr_index = get_1d_index(pp, r , p);
    //         if(basis_function_screen_data[one_d_pr_index] == false){
    //             continue;
    //         }
    //         for(int i=0; i<occ; i++){
    //             non_zero_coefficients.data()[pp][get_1d_index(non_zero_r_index, i, occ)] 
    //                 = scfdata->occupied_orbital_coefficients->data()[get_1d_index(r, i, occ)];
    //         }
    //         for(int QQ=0; QQ<Q; QQ++) {//brute force probably slow hopefully sparse dgemm can alleviate the need for this?
    //             int triangular_screened_index = screened_triangular_indicies[one_d_pr_index];
    //             B_p[get_1d_index(QQ, non_zero_r_index, non_screened_pq_indices_count[pp])]
    //                 = scfdata->three_center_integrals->data()[get_1d_index(QQ, triangular_screened_index, triangle_length)]; 
    //         }
    //         non_zero_r_index++;

    //     }
    //     M = Q;
    //     N = occ;
    //     K = non_screened_pq_indices_count[pp];
    //     lda = K; // The leading dimension of A is K when not transposed
    //     ldb = N; // The leading dimension of B is N when not transposed
    //     ldc = N; // The leading dimension of C is N

    //     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
    //     M, N, K, 
    //     alpha, B_p.data(), lda,
    //     non_zero_coefficients.data()[pp].data(), ldb,
    //     beta, 
    //     scfdata->W->data() + pp*occ*Q, ldc);

    // }
   
    // //compare the two Ws
    // int error_count = 0;
    // for (int i = 0; i < p*occ*Q; i++){
    //     if (W_new->data()[i] != scfdata->W->data()[i]){
    //         std::cout << "index: " << i << " W_new: " << W_new->data()[i] << " W: " << scfdata->W->data()[i] << std::endl;
    //         error_count++;
    //         if (error_count > 10){
    //             break;
    //         }
    //     }
    // }


    // M = p;
    // N = p;
    // K = Q*occ;
    // lda = K; // The leading dimension of A is K when not transposed
    // ldb = K; // The leading dimension of B is K when transposed
    // ldc = N; // The leading dimension of C is N
    // alpha = -1.0;
    // beta = 0.0;

    // //the dimensions of W in 3d are p x occ x Q

    // // Perform the multiplication
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, scfdata->W->data(), lda, scfdata->W->data(), ldb, beta, scfdata->K->data(), ldc);
    
    //print the top 10x10 elements of K
    std::cout << "K" << std::endl;
    for (int i = 0; i < 20; i++){
        for (int j = 0; j < 20; j++){
            std::cout << scfdata->K->data()[get_1d_index(i, j, p)] << " ";
        }
        std::cout << std::endl;
    }
}

#endif //__calculate_K_HPP__ 