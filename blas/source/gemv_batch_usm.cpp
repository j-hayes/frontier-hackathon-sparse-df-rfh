/*******************************************************************************
* Copyright 2021-2022 Intel Corporation.
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
*       This example demonstrates use of DPCPP API oneapi::mkl::blas::gemv_batch
*       using unified shared memory to perform many matrix-vectors multiplication
*       on a SYCL device (CPU, GPU).
*
*       Y[i] = alpha[i] * op[i](A[i]) * X[i] + beta[i] * Y[i]
*
*       The supported data types for gemv_batch vectors data are:
*           float
*           double
*           std::complex<float>
*           std::complex<double>
*
*******************************************************************************/

// stl includes
#include <iostream>
#include <cstdlib>
#include <limits>
#include <vector>
#include <algorithm>
#include <cstring>
#include <list>
#include <iterator>

#include <CL/sycl.hpp>
#include "oneapi/mkl/blas.hpp"
#include "mkl.h"

// local includes
#include "common_for_examples.hpp"

template <typename fp>
void run_gemv_batch_example(const sycl::device &dev) {

    //
    // Initialize data for Gemv_Batch
    //

    // number of different groups in the batch
    const std::int64_t group_count = 3;

    // Transposition operation per group
    oneapi::mkl::transpose ta[group_count] = {oneapi::mkl::transpose::T,
                                              oneapi::mkl::transpose::N,
                                              oneapi::mkl::transpose::C};
    fp alpha[group_count] = {fp{1.1}, fp{-0.4}, fp{1.0}};
    fp  beta[group_count] = {fp{-0.1}, fp{1.0}, fp{1.2}};

    // Matrix size per group
    std::int64_t          m[group_count] = {15, 19, 5};
    std::int64_t          n[group_count] = {20, 10, 13};

    // vector increment per group
    std::int64_t        lda[group_count] = {20, 19, 13};
    std::int64_t       incx[group_count] = {1, -1, 3};
    std::int64_t       incy[group_count] = {1, 2, -1};

    // batch_size per group
    std::int64_t group_size[group_count] = {50, 5, 12};

    // total batch size
    std::int64_t batch_size = 0;
    for (std::int64_t i = 0; i < group_count; i++)
        batch_size += group_size[i];

    // Catch asynchronous exceptions
    auto exception_handler = [] (sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMV_BATCH:\n"
                << e.what() << std::endl;
            }
        }
    };

    // create execution queue and buffers of matrix data
    sycl::queue main_queue(dev, exception_handler);
    sycl::event gemv_batch_done;
    std::vector<sycl::event> gemv_batch_dependencies;
    sycl::context cxt = main_queue.get_context();

    auto a = (fp **) malloc_shared(batch_size * sizeof(fp *), dev, cxt);
    auto x = (fp **) malloc_shared(batch_size * sizeof(fp *), dev, cxt);
    auto y = (fp **) malloc_shared(batch_size * sizeof(fp *), dev, cxt);
    if (!a || !x || !y)
        throw std::runtime_error("Failed to allocate USM memory.");

    std::int64_t idx = 0, sizea, sizex, sizey;
    for (std::int64_t i = 0; i < group_count; i++) {
        sizea = lda[i] * n[i];
        sizex = (ta[i] == oneapi::mkl::transpose::N) ?
            1 + (n[i] - 1) * std::abs(incx[i]) : 1 + (m[i] - 1) * std::abs(incx[i]);
        sizey = (ta[i] == oneapi::mkl::transpose::N) ?
            1 + (m[i] - 1) * std::abs(incy[i]) : 1 + (n[i] - 1) * std::abs(incy[i]);
        for (std::int64_t j = 0; j < group_size[i]; j++) {
            a[idx] = (fp *) malloc_shared(sizea * sizeof(fp), dev, cxt);
            x[idx] = (fp *) malloc_shared(sizex * sizeof(fp), dev, cxt);
            y[idx] = (fp *) malloc_shared(sizey * sizeof(fp), dev, cxt);

            if (!a[idx] || !x[idx] || !y[idx])
                throw std::runtime_error("Failed to allocate USM memory.");

            rand_matrix(a[idx], oneapi::mkl::transpose::N, m[i], n[i], lda[i]);
            if (ta[i] == oneapi::mkl::transpose::N) {
                rand_vector(x[idx], n[i], incx[i]);
                rand_vector(y[idx], m[i], incy[i]);
            }
            else {
                rand_vector(x[idx], m[i], incx[i]);
                rand_vector(y[idx], n[i], incy[i]);
            }
            idx++;
        }
    }

    //
    // Execute Gemv_Batch
    //

    // add oneapi::mkl::blas::gemv_batch to execution queue
    try {
        gemv_batch_done = oneapi::mkl::blas::gemv_batch(main_queue, ta, m, n, alpha, (const fp **) a, lda, (const fp **) x, incx, beta, y, incy, group_count, group_size, gemv_batch_dependencies);
    }
    catch(sycl::exception const& e) {
        std::cout << "\t\tCaught synchronous SYCL exception during GEMV_BATCH:\n"
                  << e.what() << std::endl << "OpenCL status: " << get_error_code(e) << std::endl;
    }

    gemv_batch_done.wait();

    //
    // Post Processing
    //

    std::cout << "\n\t\tGEMV_BATCH parameters:\n";
    std::cout << "\t\t\tgroup_count = " << group_count << std::endl;
    std::cout << "\t\t\ttranspose A = ";
    for (std::int64_t i = 0; i < group_count; i++) std::cout << (int) ta[i] << " ";
    std::cout << std::endl;
    std::cout << "\t\t\tm = ";
    for (std::int64_t i = 0; i < group_count; i++) std::cout << m[i] << " ";
    std::cout << std::endl;
    std::cout << "\t\t\tn = ";
    for (std::int64_t i = 0; i < group_count; i++) std::cout << n[i] << " ";
    std::cout << std::endl;
    std::cout << "\t\t\talpha = ";
    for (std::int64_t i = 0; i < group_count; i++) std::cout << alpha[i] << " ";
    std::cout << std::endl;
    std::cout << "\t\t\tlda = ";
    for (std::int64_t i = 0; i < group_count; i++) std::cout << lda[i] << " ";
    std::cout << std::endl;
    std::cout << "\t\t\tincx = ";
    for (std::int64_t i = 0; i < group_count; i++) std::cout << incx[i] << " ";
    std::cout << std::endl;
    std::cout << "\t\t\tincy = ";
    for (std::int64_t i = 0; i < group_count; i++) std::cout << incy[i] << " ";
    std::cout << std::endl;
    std::cout << "\t\t\tgroup_size = ";
    for (std::int64_t i = 0; i < group_count; i++) std::cout << group_size[i] << " ";
    std::cout << std::endl;

    idx = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        for (std::int64_t j = 0; j < group_size[i]; j++) {
            free(a[idx], cxt);
            free(x[idx], cxt);
            free(y[idx], cxt);
            idx++;
        }
    }
    free(a, cxt);
    free(x, cxt);
    free(y, cxt);
}

//
// Description of example setup, apis used and supported floating point type precisions
//
void print_example_banner() {

    std::cout << "" << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << "# Batch  gemv using Unified Shared Memory Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# y[i] = alpha[i] * op[i](a[i]) * x[i] + beta[i] * y[i]" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# where a is a general matrix and y and x are vectors" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   gemv_batch" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Supported data type:" << std::endl;
    std::cout << "#   float" << std::endl;
    std::cout << "#   double" << std::endl;
    std::cout << "#   std::complex<float>" << std::endl;
    std::cout << "#   std::complex<double>" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << std::endl;

}


//
// Main entry point for example.
//
// Dispatches to appropriate device types as set at build time with flag:
// -DSYCL_DEVICES_cpu -- only runs SYCL CPU device
// -DSYCL_DEVICES_gpu -- only runs SYCL GPU device
// -DSYCL_DEVICES_all (default) -- runs on all: CPU and GPU devices
//
//  For each device selected and each data type supported, Gemv_Batch Example
//  runs with all supported data types
//
int main (int argc, char ** argv) {


    print_example_banner();

    std::list<my_sycl_device_types> list_of_devices;
    set_list_of_devices(list_of_devices);

    for (auto dev_type : list_of_devices) {

        sycl::device my_dev;
        bool my_dev_is_found = false;
        get_sycl_device(my_dev, my_dev_is_found, dev_type);

        if (my_dev_is_found) {
            std::cout << "Running tests on " << sycl_device_names[dev_type] << ".\n";

            std::cout << "\tRunning with single precision real data type:" << std::endl;
            run_gemv_batch_example<float>(my_dev);

            if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0) {
                std::cout << "\tRunning with double precision real data type:" << std::endl;
                run_gemv_batch_example<double>(my_dev);
            }

            std::cout << "\tRunning with single precision complex data type:" << std::endl;
            run_gemv_batch_example<std::complex<float>>(my_dev);

            if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0) {
                std::cout << "\tRunning with double precision complex data type:" << std::endl;
                run_gemv_batch_example<std::complex<double>>(my_dev);
            }
        } else {
#ifdef FAIL_ON_MISSING_DEVICES
            std::cout << "No " << sycl_device_names[dev_type] << " devices found; Fail on missing devices is enabled.\n";
                return 1;
#else
            std::cout << "No " << sycl_device_names[dev_type] << " devices found; skipping " << sycl_device_names[dev_type] << " tests.\n";
#endif
        }


    }

    mkl_free_buffers();
    return 0;

}
