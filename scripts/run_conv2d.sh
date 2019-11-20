export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

rm -rf conv2d_result
mkdir conv2d_result

OMP_NUM_THREADS=1 numactl --physcpubind=0 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log0 &
OMP_NUM_THREADS=1 numactl --physcpubind=1 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log1 &
OMP_NUM_THREADS=1 numactl --physcpubind=2 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log2 &
OMP_NUM_THREADS=1 numactl --physcpubind=3 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log3 &
OMP_NUM_THREADS=1 numactl --physcpubind=4 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log4 &
OMP_NUM_THREADS=1 numactl --physcpubind=5 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log5 &
OMP_NUM_THREADS=1 numactl --physcpubind=6 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log6 &
OMP_NUM_THREADS=1 numactl --physcpubind=7 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log7 &
OMP_NUM_THREADS=1 numactl --physcpubind=8 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log8 &
OMP_NUM_THREADS=1 numactl --physcpubind=9 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log9 &
OMP_NUM_THREADS=1 numactl --physcpubind=10 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log10 &
OMP_NUM_THREADS=1 numactl --physcpubind=11 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log11 &
OMP_NUM_THREADS=1 numactl --physcpubind=12 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log12 &
OMP_NUM_THREADS=1 numactl --physcpubind=13 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log13 &
OMP_NUM_THREADS=1 numactl --physcpubind=14 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log14 &
OMP_NUM_THREADS=1 numactl --physcpubind=15 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log15 &
OMP_NUM_THREADS=1 numactl --physcpubind=16 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log16 &
OMP_NUM_THREADS=1 numactl --physcpubind=17 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log17 &
OMP_NUM_THREADS=1 numactl --physcpubind=18 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log18 &
OMP_NUM_THREADS=1 numactl --physcpubind=19 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log19 &
OMP_NUM_THREADS=1 numactl --physcpubind=20 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log20 &
OMP_NUM_THREADS=1 numactl --physcpubind=21 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log21 &
OMP_NUM_THREADS=1 numactl --physcpubind=22 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log22 &
OMP_NUM_THREADS=1 numactl --physcpubind=23 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log23 &
OMP_NUM_THREADS=1 numactl --physcpubind=24 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log24 &
OMP_NUM_THREADS=1 numactl --physcpubind=25 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log25 &
OMP_NUM_THREADS=1 numactl --physcpubind=26 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log26 &
OMP_NUM_THREADS=1 numactl --physcpubind=27 --membind=0 ../build/tests/benchdnn/benchdnn --conv --mode=p --cfg=u8s8s32 --dir=FWD_B --batch=../tests/benchdnn/inputs/conv/shape_conv2d_mkldnn 2>&1 | tee ./conv2d_result/log27

