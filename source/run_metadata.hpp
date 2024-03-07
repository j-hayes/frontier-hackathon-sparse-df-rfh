#ifndef __run_metadata_hpp
#define __run_metadata_hpp
#include <vector>

struct run_metadata{
    int p;
    int occ;
    int Q;
    int triangle_length;
    std::vector<int>* sparse_p_start_indices;
    std::vector<int>* screened_triangular_indicies;
    std::vector<int>* non_screened_pq_indices_count;// number of non screened pq indices for a given p
};

#endif // !