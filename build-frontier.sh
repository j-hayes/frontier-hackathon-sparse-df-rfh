# module load openblas
# module load cmake
# module load cray-hdf5
# module load rocm/6.0.0

cmake -DCMAKE_CXX_FLAGS="-I/sw/frontier/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/openblas-0.3.17-mnemgqrf4u47isw3rkt4abvto5neqef7/include -I/opt/cray/pe/hdf5/1.12.2.1/CRAYCLANG/14.0/include" -DCMAKE_EXE_LINKER_FLAGS="-L/opt/cray/pe/hdf5/1.12.2.1/CRAYCLANG/14.0/lib -lhdf5 -lhdf5_cpp" -DCMAKE_CXX_COMPILER=hipcc   ..

mkdir build
cd build
cmake ..
make -j4

# prepend_path("CMAKE_PREFIX_PATH","/sw/frontier/spack-envs/base/opt/cray-sles15-zen3/cce-15.0.0/hdf5-1.14.0-uzik26xpydtjsv32
# ju4nk7tkxa6dos5n/.")

# export CMAKE_PREFIX_PATH=/opt/cray/pe/hdf5/1.12.2.1/:$CMAKE_PREFIX_PATH