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
*       This example demonstrates use of DPCPP API oneapi::mkl::blas::gemm
*       using unified shared memory to perform General
*       Matrix-Matrix Multiplication on a SYCL devices including multi-stack deviceis.
*
*       C = alpha * op(A) * op(B) + beta * C
*
*       where op() is defined by one of oneapi::mkl::transpose::{nontrans,trans,conjtrans}
*
*
*       The supported floating point data types for gemm matrix data are:
*           float
*           double
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
// Main example for Gemm consisting of
// initialization of A, B and C matrices as well as
// scalars alpha and beta.  Then the product
//
// C = alpha * op(A) * op(B) + beta * C
//
// is performed and finally the results are post processed.
//
template <typename fp>
void run_gemm_example(const sycl::device &dev) {

    //
    // Initialize data for Gemm
    //
    // C = alpha * op(A) * op(B)  + beta * C
    //

    oneapi::mkl::transpose transA = oneapi::mkl::transpose::trans;
    oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;

    // Matrix data sizes
    int m = 45;
    int n = 98;
    int k = 67;

    // Leading dimensions of data
    int ldA = 103;
    int ldB = 105;
    int ldC = 106;

    // Set scalar fp values
    fp alpha = set_fp_value(fp(2.0), fp(-0.5));
    fp beta  = set_fp_value(fp(3.0), fp(-1.5));

    // Create sub devices for multi-stack device
    std::vector<sycl::device> subdev = {};
    sycl::queue subdev_queue0;
    sycl::queue subdev_queue1;
    try {
      using namespace sycl::info;
      subdev = dev.create_sub_devices<
          partition_property::partition_by_affinity_domain>(
          partition_affinity_domain::numa);
      std::cout << "\t\tUse " << subdev.size() <<" subdevices" << std::endl;
    } catch (sycl::exception &e) {
        std::cout << "\t\tSubdevices not found, use whole device" << std::endl;
    }

    int nb_device = subdev.size();

    // Catch asynchronous exceptions
    auto exception_handler = [] (sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM:\n"
                << e.what() << std::endl;
            }
        }
    };

    // Create context, execution queue, and buffers of matrix data
    subdev.push_back(dev);
    sycl::context cxt(subdev);
    sycl::queue main_queue(cxt, dev, exception_handler);

    if (nb_device > 1) {
        subdev_queue0 = sycl::queue(cxt, subdev[0], exception_handler);
        subdev_queue1 = sycl::queue(cxt, subdev[1], exception_handler);
    }
    sycl::event gemm_done0;
    sycl::event gemm_done1;
    std::vector<sycl::event> gemm_dependencies;

    int sizea = (transA == oneapi::mkl::transpose::nontrans) ? ldA * k : ldA * m;
    int sizeb = ldB * n;
    int sizec = ldC * n;

    auto A = (fp *) malloc_shared(sizea * sizeof(fp), dev, cxt);
    auto B = (fp *) malloc_shared(sizeb * sizeof(fp), dev, cxt);
    auto C = (fp *) malloc_shared(sizec * sizeof(fp), dev, cxt);

    if (!A || !B || !C)
        throw std::runtime_error("Failed to allocate USM memory.");

    rand_matrix(A, transA, m, k, ldA);
    rand_matrix(B, transB, k, n, ldB);
    rand_matrix(C, oneapi::mkl::transpose::nontrans, m, n, ldC);

    //
    // Execute GEMM
    //

    try {
        if (nb_device > 1) {
            // Split B and C for subdevices
            int n_subdev = n/2;
            int sizeb_subdev = ldB * n_subdev;
            int sizec_subdev = ldC * n/2;

            gemm_done0 = oneapi::mkl::blas::gemm(subdev_queue0, transA, transB, m, n_subdev, k, alpha, A, ldA, B, ldB, beta, C, ldC, gemm_dependencies);
            gemm_done1 = oneapi::mkl::blas::gemm(subdev_queue1, transA, transB, m, n_subdev, k, alpha, A, ldA, B + sizeb_subdev, ldB, beta, C + sizec_subdev, ldC, gemm_dependencies);
        } else {
            gemm_done0 = oneapi::mkl::blas::gemm(main_queue, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, gemm_dependencies);
        }
    }
    catch(sycl::exception const& e) {
        std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
                  << e.what() << std::endl << "OpenCL status: " << get_error_code(e) << std::endl;
    }

    gemm_done0.wait();
    if (nb_device > 1) gemm_done1.wait();

    //
    // Post Processing
    //

    std::cout << "\n\t\tGEMM parameters:\n";
    std::cout << "\t\t\ttransA = " << ( transA == oneapi::mkl::transpose::nontrans ? "nontrans" : ( transA == oneapi::mkl::transpose::trans ? "trans" : "conjtrans"))
              <<   ", transB = " << ( transB == oneapi::mkl::transpose::nontrans ? "nontrans" : ( transB == oneapi::mkl::transpose::trans ? "trans" : "conjtrans")) << std::endl;
    std::cout << "\t\t\tm = " << m << ", n = " << n << ", k = " << k << std::endl;
    std::cout << "\t\t\tlda = " << ldA << ", ldB = " << ldB << ", ldC = " << ldC << std::endl;
    std::cout << "\t\t\talpha = " << alpha << ", beta = " << beta << std::endl;


    std::cout << "\n\t\tOutputting 2x2 block of A,B,C matrices:" << std::endl;

    // output the top 2x2 block of A matrix
    print_2x2_matrix_values(A, ldA, "A");

    // output the top 2x2 block of B matrix
    print_2x2_matrix_values(B, ldB, "B");

    // output the top 2x2 block of C matrix
    print_2x2_matrix_values(C, ldC, "C");

    free(A, cxt);
    free(B, cxt);
    free(C, cxt);
}

//
// Description of example setup, apis used and supported floating point type precisions
//
void print_example_banner() {

    std::cout << "" << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << "# General Matrix-Matrix Multiplication using Unified Shared Memory Example " << std::endl;
    std::cout << "# with multi-stack devices support:" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# C = alpha * A * B + beta * C" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# where A, B and C are general dense matrices and alpha, beta are" << std::endl;
    std::cout << "# floating point type precision scalars." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   gemm" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Supported floating point type precisions:" << std::endl;
    std::cout << "#   float" << std::endl;
    std::cout << "#   double" << std::endl;
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
//  For each device selected and each data type supported, Gemm Example
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
            run_gemm_example<float>(my_dev);

            if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0) {
                std::cout << "\tRunning with double precision real data type:" << std::endl;
                run_gemm_example<double>(my_dev);
            }

            std::cout << "\tRunning with single precision complex data type:" << std::endl;
            run_gemm_example<std::complex<float>>(my_dev);

            if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0) {
                std::cout << "\tRunning with double precision complex data type:" << std::endl;
                run_gemm_example<std::complex<double>>(my_dev);
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
