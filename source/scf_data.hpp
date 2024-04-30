#ifndef __SCF_DATA_HPP__
#define __SCF_DATA_HPP__

#include <vector>

struct scf_data {
    std::vector<double>* J;
    std::vector<double>* V;
    std::vector<double>* J_AB_inv;
    std::vector<double>* density;
    std::vector<double>* occupied_orbital_coefficients;
    std::vector<double>* two_center_integrals;
    std::vector<double>* three_center_integrals;
    std::vector<double>* three_center_integrals_T;
    std::vector<double>* K;
    std::vector<double>* W;
    std::vector<double>* two_electron_fock;
    std::vector<std::vector<double>>* non_zero_coefficients;

};

#endif // !1