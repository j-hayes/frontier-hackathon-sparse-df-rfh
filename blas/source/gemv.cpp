/*******************************************************************************
* Copyright 2018-2022 Intel Corporation.
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
*       This example demonstrates use of DPCPP API oneapi::mkl::blas::gemv to perform General
*       Matrix-Vector Multiplication on a SYCL device (CPU, GPU).
*
*       y = alpha * op(A) * x + beta * y
*
*       where op() is defined by one of oneapi::mkl::transpose::{nontrans,trans,conjtrans}
*
*
*       The supported floating point data types for gemv matrix/vector data are:
*           float
*           double
*           std::complex<float>
*           std::complex<double>
*
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


//
// Main example for Gemv consisting of
// initialization of x and y vectors, A matrix as well as
// scalars alpha and beta.  Then the product
//
// y = alpha * op(A) * x + beta * y
//
// is performed and finally the results are post processed.
//
template <typename fp>
void run_gemv_example(const sycl::device &dev) {

    //
    // Initialize data for Gemv
    //
    // y = alpha * op(A) * x + beta * y
    //

    oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;

    // matrix data sizes
    int m = 79;
    int n = 83;

    // leading dimension of data
    int ldA = 103;

    // increment for x and y
    int incx = 2;
    int incy = 3;

    // set scalar fp values
    fp alpha = set_fp_value(fp(2.0), fp(-0.5));
    fp beta  = set_fp_value(fp(3.0), fp(-1.5));

    // prepare matrix data
    std::vector <fp, mkl_allocator<fp, 64>> A;

    // prepare vector data
    std::vector <fp, mkl_allocator<fp, 64>> x;
    std::vector <fp, mkl_allocator<fp, 64>> y;

    int x_len = outer_dimension(transA, m, n);
    int y_len = inner_dimension(transA, m, n);

    rand_matrix(A, transA, m, n, ldA);
    rand_vector(x, x_len, incx);
    rand_vector(y, y_len, incy);


    //
    // Execute Gemv
    //

    // Catch asynchronous exceptions
    auto exception_handler = [] (sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMV:\n"
                << e.what() << std::endl;
            }
        }
    };

    // create execution queue and buffers of matrix data
    sycl::queue main_queue(dev, exception_handler);

    sycl::buffer<fp, 1> A_buffer(A.data(), A.size());
    sycl::buffer<fp, 1> x_buffer(x.data(), x.size());
    sycl::buffer<fp, 1> y_buffer(y.data(), y.size());

    // add oneapi::mkl::blas::gemv to execution queue
    try {
        oneapi::mkl::blas::gemv(main_queue, transA, m, n, alpha, A_buffer, ldA, x_buffer, incx, beta, y_buffer, incy);
    }
    catch(sycl::exception const& e) {
        std::cout << "\t\tCaught synchronous SYCL exception during GEMV:\n"
                  << e.what() << std::endl << "OpenCL status: " << get_error_code(e) << std::endl;
    }


    //
    // Post Processing
    //

    std::cout << "\n\t\tGEMV parameters:\n";
    std::cout << "\t\t\ttransA = " << ( transA == oneapi::mkl::transpose::nontrans ? "nontrans" : ( transA == oneapi::mkl::transpose::trans ? "trans" : "conjtrans")) << std::endl;
    std::cout << "\t\t\tm = " << m << ", n = " << n << std::endl;
    std::cout << "\t\t\tlda = " << ldA << std::endl;
    std::cout << "\t\t\tincx = " << incx << ", incy = " << incy << std::endl;
    std::cout << "\t\t\talpha = " << alpha << ", beta = " << beta << std::endl;


    std::cout << "\n\t\tOutputting 2x2 block of A matrix and 2x1 blocks of x and y vectors:" << std::endl;

    // output the top 2x2 block of A matrix
    auto A_accessor = A_buffer.template get_access<sycl::access::mode::read>();
    print_2x2_matrix_values(A_accessor, ldA, "A");

    // output the top 2x1 block of x vector
    auto x_accessor = x_buffer.template get_access<sycl::access::mode::read>();
    print_2x1_vector_values(x_accessor, incx, "x");

    // output the top 2x1 block of y vector
    auto y_accessor = y_buffer.template get_access<sycl::access::mode::read>();
    print_2x1_vector_values(y_accessor, incy, "y");


}

//
// Description of example setup, apis used and supported floating point type precisions
//
void print_example_banner() {

    std::cout << "" << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << "# General Matrix-Vector Multiplication Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# y = alpha * A * x + beta * y" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# where A is general dense matrix, x and y are general dense vectors," << std::endl;
    std::cout << "# and alpha and beta are floating point type precision scalars." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   gemv" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Supported floating point type precisions:" << std::endl;
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
//  For each device selected and each data type supported, Gemv Example
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
            run_gemv_example<float>(my_dev);

            if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0) {
                std::cout << "\tRunning with double precision real data type:" << std::endl;
                run_gemv_example<double>(my_dev);
            }

            std::cout << "\tRunning with single precision complex data type:" << std::endl;
            run_gemv_example<std::complex<float>>(my_dev);

            if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0) {
                std::cout << "\tRunning with double precision complex data type:" << std::endl;
                run_gemv_example<std::complex<double>>(my_dev);
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
