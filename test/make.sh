#GPU_ARCH=sm_13
GPU_ARCH=sm_121
#CUDA_SDK_HOME=/home/mian/NVIDIA_GPU_Computing_SDK
CUDA_SDK_HOME=/home/tkouya/NVIDIA_CUDA-6.5_Samples/common

echo "Compiling GPU kernel ......"
nvcc gqdtest_kernel.cu -c -fmad=false -O3 -I../inc -I ../map -I$CUDA_SDK_HOME/inc -arch=$GPU_ARCH
echo "Compiling test cases ......"
g++ test_util.cpp -c -O3 -I ../inc -I /usr/local/cuda/include -fopenmp
g++ benchmark.cpp -c -O3 -I../inc -I/usr/local/cuda/include -fopenmp
echo "Linking ......"
g++ test_util.o gqdtest_kernel.o benchmark.o -o benchmark -O3 -L$CUDA_SDK_HOME/lib -L/usr/local/cuda/lib64  -lqd -lcuda -lcudart -fopenmp
