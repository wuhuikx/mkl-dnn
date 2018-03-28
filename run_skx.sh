sudo echo 0 > /proc/sys/kernel/numa_balancing
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export KMP_AFFINITY="granularity=fine,compact,0,0"
#export KMP_AFFINITY="explicit,1,0,granularity=fine, proclist=[28-55]"
#cpupower frequency-set -g performance 

cd build
make -j `nproc`
cd ..

#./build/tests/gtests/test_convolution_relu_forward_u8s8s32>log.txt 2>1&

#./build/tests/gtests/test_convolution_relu_forward_u8s8s32>log1.txt 2>1&

#./build/tests/gtests/test_convolution_relu_forward_u8s8s32>log2.txt 2>1&

#nohup ./build/tests/gtests/test_convolution_relu_forward_u8s8s32>log3.txt 2>1&

#taskset -c 28-55 numactl -l ./build/tests/gtests/test_convolution_relu_forward_u8s8s32
taskset -c 0-27 numactl -l ./build/tests/gtests/test_convolution_relu_forward_u8s8s32

echo "test vnni forward done"

#/home/cldai/package/sde/sde-external-8.12.0-2017-10-23-lin/sde -- ./build/tests/gtests/test_convolution_relu_forward_u8s8s32

#echo "conv relu fuse done"

#/home/cldai/package/sde/sde-external-8.12.0-2017-10-23-lin/sde -- ./build/tests/gtests/test_convolution_relu_forward_u8s8s32_discrete

#echo "conv relu non-fuse done"
