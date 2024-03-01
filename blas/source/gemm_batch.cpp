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
*       This example demonstrates use of DPCPP API oneapi::mkl::blas::gemm_batch
*       using unified shared memory to perform batched General
*       Matrix-Matrix Multiplication on a SYCL device (CPU, GPU).
*
*       Ci = alphai * opi(Ai) * opi(Bi) + betai * Ci
*
*       where opi() is defined by one of oneapi::mkl::transpose::{nontrans,trans,conjtrans}
*
*
*       The supported floating point data types for gemm_batch matrix data are:
*           sycl::half
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


//
// Main example for gemm_batch consisting of
// initialization of Ai, Bi and Ci matrices as well as
// scalars alphai and betai.  Then the product
//
// Ci = alphai * op(Ai) * op(Bi) + betai * Ci
//
// is performed and finally the results are post processed.
//
template <typename fp>
void run_gemm_batch_example(const sycl::device &dev) {

    //
    // Initialize data for Gemm batch 1 group,
    // all GEMM parameter are constant except the scalars and matrices
    //
    // Ci = alphai * op(Ai) * op(Bi)  + betai * Ci
    //

    size_t batch_size = 10;

    // Catch asynchronous exceptions
    auto exception_handler = [] (sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during batched GEMM:\n"
                << e.what() << std::endl;
            }
        }
    };
    sycl::queue main_queue(dev, exception_handler);
    sycl::event gemm_batch_done;
    std::vector<sycl::event> gemm_batch_dependencies;

    sycl::span ta(sycl::malloc_shared<oneapi::mkl::transpose>(1, main_queue), 1);
    sycl::span tb(sycl::malloc_shared<oneapi::mkl::transpose>(1, main_queue), 1);

    if (!ta.data() || !tb.data())
        throw std::runtime_error("Failed to allocate USM memory.");

    sycl::span m(sycl::malloc_shared<std::int64_t>(1, main_queue), 1);
    sycl::span n(sycl::malloc_shared<std::int64_t>(1, main_queue), 1);
    sycl::span k(sycl::malloc_shared<std::int64_t>(1, main_queue), 1);

    if (!m.data() || !n.data() || !k.data())
        throw std::runtime_error("Failed to allocate USM memory.");

    sycl::span lda(sycl::malloc_shared<std::int64_t>(1, main_queue), 1);
    sycl::span ldb(sycl::malloc_shared<std::int64_t>(1, main_queue), 1);
    sycl::span ldc(sycl::malloc_shared<std::int64_t>(1, main_queue), 1);

    if (!lda.data() || !ldb.data() || !ldc.data())
        throw std::runtime_error("Failed to allocate USM memory.");

    sycl::span group_size(sycl::malloc_shared<size_t>(1, main_queue), 1);

    if (!group_size.data())
        throw std::runtime_error("Failed to allocate USM memory.");

    sycl::span alpha(sycl::malloc_shared<fp>(batch_size, main_queue), batch_size);
    sycl::span beta(sycl::malloc_shared<fp>(batch_size, main_queue), batch_size);

    if (!alpha.data() || !beta.data())
        throw std::runtime_error("Failed to allocate USM memory.");

    auto a = sycl::malloc_shared<fp*>(batch_size, main_queue);
    auto b = sycl::malloc_shared<fp*>(batch_size, main_queue);
    sycl::span c(sycl::malloc_shared<fp*>(batch_size, main_queue), batch_size);

    if (!a || !b || !c.data())
        throw std::runtime_error("Failed to allocate USM memory.");

    ta[0] = oneapi::mkl::transpose::trans;
    tb[0] = oneapi::mkl::transpose::nontrans;
    m[0] = 45; n[0] = 98; k[0] = 67;
    lda[0] = 103; ldb[0] = 105; ldc[0] = 106;
    group_size[0] = batch_size;

    std::int64_t sizea, sizeb, sizec = ldc[0] * n[0];

    sizea = (ta[0] == oneapi::mkl::transpose::nontrans) ? lda[0] * k[0] : lda[0] * m[0];
    sizeb = (tb[0] == oneapi::mkl::transpose::nontrans) ? ldb[0] * n[0] : ldb[0] * k[0];

    for (std::int64_t i = 0; i < batch_size; i++) {
        a[i] = sycl::malloc_shared<fp>(sizea, main_queue);
        b[i] = sycl::malloc_shared<fp>(sizeb, main_queue);
        c[i] = sycl::malloc_shared<fp>(sizec, main_queue);

        if (!a[i] || !b[i] || !c[i])
            throw std::runtime_error("Failed to allocate USM memory.");

        rand_matrix(a[i], ta[0], m[0], k[0], lda[0]);
        rand_matrix(b[i], tb[0], k[0], n[0], ldb[0]);
        rand_matrix(c[i], oneapi::mkl::transpose::nontrans, m[0], n[0], ldc[0]);
        alpha[i] = rand_scalar<fp>();
        beta[i] = rand_scalar<fp>();
    }

    //
    // Execute gemm_batch
    //

    // add oneapi::mkl::blas::gemm_batch to execution queue
    try {
        gemm_batch_done = oneapi::mkl::blas::gemm_batch(main_queue, ta, tb, m, n, k, alpha,
                                                        {(const fp**) a, (size_t) batch_size}, lda,
                                                        {(const fp**) b, (size_t) batch_size}, ldb, beta, c, ldc,
                                                        1, group_size, gemm_batch_dependencies);
    }
    catch(sycl::exception const& e) {
        std::cout << "\t\tCaught synchronous SYCL exception during batched GEMM:\n"
                  << e.what() << std::endl << "OpenCL status: " << get_error_code(e) << std::endl;
    }
    catch(...) {
        std::cout << "some exceptions are raised\n";
    }

    gemm_batch_done.wait();

    //
    // Post Processing
    //

    std::cout << "\n\t\tGEMM batch parameters:\n";
    std::cout << "\t\t\ttransA = " << ( ta[0] == oneapi::mkl::transpose::nontrans ? "nontrans" : ( ta[0] == oneapi::mkl::transpose::trans ? "trans" : "conjtrans"))
              <<   ", transB = " << ( tb[0] == oneapi::mkl::transpose::nontrans ? "nontrans" : ( tb[0] == oneapi::mkl::transpose::trans ? "trans" : "conjtrans")) << std::endl;
    std::cout << "\t\t\tm = " << m[0] << ", n = " << n[0] << ", k = " << k[0] << std::endl;
    std::cout << "\t\t\tlda = " << lda[0] << ", ldB = " << ldb[0] << ", ldC = " << ldc[0] << std::endl;
    std::cout << "\t\t\talpha = ";
    for (std::int64_t i = 0; i < batch_size; i++)
        std::cout << alpha[i] << " ";
    std::cout << "\n\t\t\tbeta = ";
    for (std::int64_t i = 0; i < batch_size; i++)
        std::cout << beta[i] << " ";
    std::cout << std::endl;

    std::cout << "\n\t\tOutputting 2x2 block of first A,B,C matrices:" << std::endl;

    // output the top 2x2 block of A matrix
    print_2x2_matrix_values(a[0], lda[0], "A");

    // output the top 2x2 block of B matrix
    print_2x2_matrix_values(b[0], ldb[0], "B");

    // output the top 2x2 block of C matrix
    print_2x2_matrix_values(c[0], ldc[0], "C");

    for (int i = 0; i < batch_size; i++) {
        sycl::free(a[i], main_queue);
        sycl::free(b[i], main_queue);
        sycl::free(c[i], main_queue);
    }

    sycl::free(ta.data(), main_queue);
    sycl::free(tb.data(), main_queue);
    sycl::free(m.data(), main_queue);
    sycl::free(n.data(), main_queue);
    sycl::free(k.data(), main_queue);
    sycl::free(lda.data(), main_queue);
    sycl::free(ldb.data(), main_queue);
    sycl::free(ldc.data(), main_queue);
    sycl::free(group_size.data(), main_queue);
    sycl::free(alpha.data(), main_queue);
    sycl::free(beta.data(), main_queue);
    sycl::free(a, main_queue);
    sycl::free(b, main_queue);
    sycl::free(c.data(), main_queue);
}

//
// Description of example setup, apis used and supported floating point type precisions
//
void print_example_banner() {

    std::cout << "" << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << "# General batched Matrix-Matrix Multiplication using Unified Shared Memory Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Ci = alphai * Ai * Bi + betai * Ci" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# where Ai, Bi and Ci are general dense matrices and alphai, betai are" << std::endl;
    std::cout << "# floating point type precision scalars." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   gemm_batch" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Supported floating point type precisions:" << std::endl;
    std::cout << "#   sycl::half" << std::endl;
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
//  For each device selected and each data type supported, Gemm batch Example
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

            std::cout << "\tRunning with half precision real data type:" << std::endl;
            run_gemm_batch_example<sycl::half>(my_dev);

            std::cout << "\tRunning with single precision real data type:" << std::endl;
            run_gemm_batch_example<float>(my_dev);

            if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0) {
                std::cout << "\tRunning with double precision real data type:" << std::endl;
                run_gemm_batch_example<double>(my_dev);
            }

            std::cout << "\tRunning with single precision complex data type:" << std::endl;
            run_gemm_batch_example<std::complex<float>>(my_dev);

            if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0) {
                std::cout << "\tRunning with double precision complex data type:" << std::endl;
                run_gemm_batch_example<std::complex<double>>(my_dev);
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
