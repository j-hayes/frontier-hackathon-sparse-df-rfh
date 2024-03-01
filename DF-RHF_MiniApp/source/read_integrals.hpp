#ifndef __DF_read_integrals__
#define __DF_read_integrals__


#include <iostream>
#include <string>
#include "H5Cpp.h"

using namespace H5;


// #ifdef OLD_HEADER_FILENAME
// #include <iostream.h>
// #else
// #include <iostream>
// #endif
// #include <string>
// #ifndef H5_NO_NAMESPACE
// #ifndef H5_NO_STD
//     // using std::cout;
//     // using std::endl;
// #endif  // H5_NO_STD
// #endif
// #include "H5Cpp.h"
// #ifndef H5_NO_NAMESPACE
//     using namespace H5;
// #endif
// // const H5std_string FILE_NAME( "/home/jacksonjhayes/source/mklexamples/dpcpp/DF-RHF_MiniApp/data/basis_function_screen_matrix.h5" );
// // const H5std_string DATASET_NAME( "IntArray" );
// // const int    NX_SUB = 3;        // hyperslab dimensions
// // const int    NY_SUB = 4;
// // const int    NX = 7;            // output buffer dimensions
// // const int    NY = 7;
// // const int    NZ = 3;
// // const int    RANK_OUT = 3;




// void read_hdf5_data(){
//     //    /*
//     //    * Turn off the auto-printing when failure occurs so that we can
//     //    * handle the errors appropriately
//     //    */
//     //   Exception::dontPrint();
//     //   /*
//     //    * Open the specified file and the specified dataset in the file.
//     //    */
//     //   H5File file( FILE_NAME, H5F_ACC_RDONLY );
//     //   DataSet dataset = file.openDataSet( DATASET_NAME );
//     //   /*
//     //    * Get the class of the datatype that is used by the dataset.
//     //    */
//     //   H5T_class_t type_class = dataset.getTypeClass();
//     //   /*
//     //    * Get class of datatype and print message if it's an integer.
//     //    */
//     //   if( type_class == H5T_INTEGER )
//     //   {
//     //      std::cout << "Data set has INTEGER type" << std::endl;
//     //      /*
//     //       * Get the integer datatype
//     //       */
//     //      IntType intype = dataset.getIntType();
//     //      /*
//     //       * Get order of datatype and print message if it's a little endian.
//     //       */
//     //      H5std_string order_string;
//     //      H5T_order_t order = intype.getOrder( order_string );
//     //      std::cout << order_string << std::endl;
//     //      /*
//     //       * Get size of the data element stored in file and print it.
//     //       */
//     //      size_t size = intype.getSize();
//     //      std::cout << "Data size is " << size << std::endl;
//     //   }

//     std::cout << "Reading HDF5 data" << std::endl;


// }

// //path: the path to the integral file
// void read_3eri_integrals(const char * path) {
//     read_hdf5_data();
// }

#endif // __DF_read_integrals__
