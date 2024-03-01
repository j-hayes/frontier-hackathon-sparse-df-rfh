#ifndef __read_hdf5_file__
#define __read_hdf5_file__


#include "H5Cpp.h"
#include <iostream>
#include <string>
using namespace H5;

template <typename T>
struct hdf5_data_with_dims
{
    T* data;
    int M;
    int N;    
};

const H5std_string DATASET_NAME("data/values");

template <typename T>
hdf5_data_with_dims<T> read_file_data(std::string const &file_name) {
     // Open the HDF5 file
     

    H5::H5File file(file_name, H5F_ACC_RDONLY);


    // Open the dataset
    H5::DataSet dataset = file.openDataSet("data/values");

    // Get the dataspace of the dataset
    H5::DataSpace dataspace = dataset.getSpace();

    // Get the number of dimensions in the dataspace
    int ndims = dataspace.getSimpleExtentNdims();

    // Get the size of each dimension
    hsize_t dims[ndims];
    dataspace.getSimpleExtentDims(dims, NULL);

    // Allocate array to hold the data
    
    hdf5_data_with_dims<T> hdf5_data;
    hdf5_data.data = new T[dims[0] * dims[1]];
    hdf5_data.M = dims[0];
    hdf5_data.N = dims[1];

    // get data type 

    // if type T is integer  from template 

    if (std::is_same<T, int>::value) {
        dataset.read(hdf5_data.data, H5::PredType::NATIVE_INT);
    }
    else if (std::is_same<T, float>::value) {
        dataset.read(hdf5_data.data, H5::PredType::NATIVE_FLOAT);
    }
    else if (std::is_same<T, double>::value) {
        dataset.read(hdf5_data.data, H5::PredType::NATIVE_DOUBLE);
    }
    else {
        throw std::runtime_error("Unsupported data type for hdf5 read");
    }
    


    


    return hdf5_data;
}

#endif // __read_hdf5_file__