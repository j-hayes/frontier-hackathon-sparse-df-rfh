# module load PrgEnv-cray
# module load cmake
# module load cray-hdf5

# cmake -DCMAKE_CXX_FLAGS="-fopenmp -I/global/u2/j/jhayes1/source/amd-utils/include -I/global/u2/j/jhayes1/source/amd-libflame/include/ILP64 -I/global/u2/j/jhayes1/source/amd-blis/include/ILP64 -I/opt/cray/pe/hdf5/1.12.2.9/gnu/12.3/include" -DCMAKE_EXE_LINKER_FLAGS="-L/global/u2/j/jhayes1/source/amd-utils/lib -laoclutils -L/global/u2/j/jhayes1/source/amd-libflame/lib/ILP64 -lflame -L/global/u2/j/jhayes1/source/amd-blis/lib/ILP64/ -lblis -L/opt/cray/pe/hdf5/1.12.2.9/gnu/12.3/lib -lhdf5 -lhdf5_cpp" -DCMAKE_CXX_COMPILER=crayCC ..

mkdir build
cmake -DCMAKE_CXX_FLAGS="-I/opt/cray/pe/libsci/23.12.5/CRAYCLANG/17.0/x86_64/include/ -I/opt/cray/pe/hdf5/1.12.2.9/gnu/12.3/include" -DCMAKE_EXE_LINKER_FLAGS="-L/opt/cray/pe/libsci/23.12.5/CRAYCLANG/17.0/x86_64/lib -lsci_cray -L/opt/cray/pe/hdf5/1.12.2.9/gnu/12.3/lib -lhdf5 -lhdf5_cpp" -DCMAKE_CXX_COMPILER=crayCC ..
cd build
cmake ..
make -j4





# prepend_path("CMAKE_PREFIX_PATH","/sw/frontier/spack-envs/base/opt/cray-sles15-zen3/cce-15.0.0/hdf5-1.14.0-uzik26xpydtjsv32
# ju4nk7tkxa6dos5n/.")

# export CMAKE_PREFIX_PATH=/opt/cray/pe/hdf5/1.12.2.1/:$CMAKE_PREFIX_PATH

export LD_LIBRARY_PATH=/opt/cray/pe/libsci/23.12.5/CRAYCLANG/17.0/x86_64/include/:$LD_LIBRARY_PATH
