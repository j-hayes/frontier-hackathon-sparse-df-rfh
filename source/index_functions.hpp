
#ifndef __INDEX_FUNCTIONS_HPP__
#define __INDEX_FUNCTIONS_HPP__


// get the 1d index for a 2d array in row major order for size MxN
int get_1d_index(int i, int j, int N){
    return (i)*N + j;
}

//get the 1d index for a 2d array in column major order for size MxN
int get_1d_index_col_major(int i, int j, int M){
    return (j)*M + i;
}

//get the 1d index for a 3d array in row major order for size MxNxO
int get_1d_index(int i, int j, int k, int N, int O){
    return (i)*N*O + (j)*O + k;
}

#endif //__INDEX_FUNCTIONS_HPP__