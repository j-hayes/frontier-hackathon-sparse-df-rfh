/*******************************************************************************
* Copyright 2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
*
*  Content:
*       This example demonstrates use of DPC++ API  Data Fitting functionality.
*       Common workflow with linear spline construction and data
*       interpolation is performed on a SYCL device (CPU, GPU).
*
*       Supported data types:
*           float
*           double
*
*
*******************************************************************************/

// STL includes
#include <cstdint>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// SYCL include
#include <CL/sycl.hpp>

// oneMKL include
#include "oneapi/mkl/experimental/data_fitting.hpp"

// Local includes
#include "oneapi/mkl.hpp" // extra for this example, but needs for common_for_examples.hpp
#include "common_for_examples.hpp"

namespace df = oneapi::mkl::experimental::data_fitting;

// Spline parameters
constexpr std::int64_t nx = 100000;
constexpr std::int64_t ny = 5;
// Quantity of interpolation sites
constexpr std::int64_t nsites = 150000;


// check spline values are equal to function values for partition points
template <typename FpType>
bool check_spline_values(FpType* coeff, FpType* functions, double epsilon){
    for (std::int64_t i = 0; i < ny; ++i) {
        for (std::int64_t j = 0; j < nx - 1; ++j) {

            if(std::abs(coeff[2 * (i * (nx - 1) + j)] - functions[i * nx + j]) > epsilon){
                std::cout<<"check_spline_values failed. i_function = "<<i<<
                    ", i_partition = "<<j<<
                    ", got = "<<coeff[2 * (i * (nx - 1) + j)]<<
                    ", expected = "<<functions[i * nx + j]<<
                std::endl;
                return false;
            }
        }
    }
    return true;
}

// check interpolation results
template <typename FpType>
bool check_interpolation_results(FpType* results, FpType* partitions, FpType* sites, FpType* coeff, double epsilon){
    for (std::int64_t j = 0; j < nsites; ++j) {

        // find correct interpolation interval
        auto cell = std::upper_bound(partitions, partitions + nx, sites[j]) - partitions;

        std::uint64_t interval;
        if (cell == 0)
            interval = 0;
        else if (cell == nx)
            interval = nx - 2;
        else if(cell > 0)
            interval = cell - 1;

        FpType t = sites[j] - partitions[interval];

        for (std::int64_t i = 0; i < ny; ++i) {
            // Get polynomial coefficients in current bin
            FpType c0 = coeff[2 * (i * (nx - 1) + interval) + 0];
            FpType c1 = coeff[2 * (i * (nx - 1) + interval) + 1];
            FpType result = c0 + t * c1;

            if(std::abs(result - results[i * nsites + j]) > epsilon){
                std::cout<<"check_interpolation_results failed. i_function = "<<i<<
                    ", i_site = "<<j<<
                    ", got = "<<result<<
                    ", expected = "<<results[i * nsites + j]<<
                std::endl;
                return false;
            }
        }
    }
    return true;
}

template <typename FpType>
bool run_example(sycl::device& dev) {

    sycl::queue q(dev);

    sycl::usm_allocator<FpType, sycl::usm::alloc::shared> alloc(q);
    // Allocate memory for spline parameters
    std::vector<FpType, decltype(alloc)> partitions(nx, alloc);
    std::vector<FpType, decltype(alloc)> functions(nx * ny, alloc);
    // 4 stands for quantity of spline coefficients per 1 interval
    std::vector<FpType, decltype(alloc)> coeffs(ny * (nx - 1) * 2, alloc);

    // Fill parameters with valid data
    for (std::int64_t i = 0; i < nx; ++i) {
        partitions[i] = 0.1f * i;
    }

    for (std::int64_t i = 0; i < ny; ++i) {
        auto freq = (i + 1) * 0.03f * 6.28f;
        for (std::int64_t j = 0; j < nx; ++j) {
            functions[i * nx + j] = std::sin(freq * partitions[j]);
        }
    }

    // Set parameters to spline
    df::spline<FpType, df::linear_spline::default_type> spl(q, ny);
    spl.set_partitions(partitions.data(), nx)
        .set_coefficients(coeffs.data())
        .set_function_values(functions.data());

    // Perform spline construction
    auto event = spl.construct();

    // Allocate memory for interpolation
    std::vector<FpType, decltype(alloc)> sites(nsites, alloc);
    std::vector<FpType, decltype(alloc)> results(nsites * ny, alloc);
    for (int i = 0; i < nsites; ++i) {
        sites[i] = (0.1f * nx * i) / nsites;
    }
    event = df::interpolate(spl, sites.data(), nsites, results.data(), { event });
    event.wait();

    return
        check_spline_values(coeffs.data(), functions.data(), 1e-6) &&
        check_interpolation_results(results.data(), partitions.data(), sites.data(), coeffs.data(), 1e-6);
}

//
// Description of example setup, APIs used and supported floating point type precisions
//
void print_example_banner() {

    std::cout << "" << std::endl;
    std::cout << "#############################################################################" << std::endl;
    std::cout << "# Example of common workflow with spline construction and data interpolation:" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using APIs:" << std::endl;
    std::cout << "#   spline interpolate" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Supported precisions:" << std::endl;
    std::cout << "#   float double" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << std::endl;

}

//
// Main entry point for example.
//
// Dispatches to appropriate device types as set at build time with flag:
// -DSYCL_DEVICES_cpu -- only runs SYCL CPU implementation
// -DSYCL_DEVICES_gpu -- only runs SYCL GPU implementation
// -DSYCL_DEVICES_all (default) -- runs on all: cpu and gpu devices
//

int main (int argc, char ** argv) {

    print_example_banner();

    std::list<my_sycl_device_types> list_of_devices;
    set_list_of_devices(list_of_devices);

    for (auto it = list_of_devices.begin(); it != list_of_devices.end(); ++it) {

        sycl::device my_dev;
        bool my_dev_is_found = false;
        get_sycl_device(my_dev, my_dev_is_found, *it);

            if(my_dev_is_found) {
                std::cout << "Running tests on " << sycl_device_names[*it] << ".\n";

                std::cout << "\tRunning with single precision real data type:" << std::endl;
                if(!run_example<float>(my_dev)) {
                    std::cout << "FAILED" << std::endl;
                    return 1;
                }
            } else {
#ifdef FAIL_ON_MISSING_DEVICES
                std::cout << "No " << sycl_device_names[*it] << " devices found; Fail on missing devices is enabled.\n";
                std::cout << "FAILED" << std::endl;
                return 1;
#else
                std::cout << "No " << sycl_device_names[*it] << " devices found; skipping " << sycl_device_names[*it] << " tests.\n";
#endif
            }
    }
    std::cout << "PASSED" << std::endl;
    return 0;
}
