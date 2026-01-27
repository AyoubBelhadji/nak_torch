#!/bin/sh

LR=0.1
N_STEPS=500
DIM=2
PROBLEM=joker
ALGORITHMS=cbs,msip_fredholm,svgd
N_ITER=10
N_PARTICLES=10,20,40,80,160,320,640
INVERSE_TEMP=0.9 # For CBS
TEST_KERNEL=sqexp
TEST_KERNEL_BANDWIDTH=1.
GRADIENT_DECAY=0.9
KERNEL_DIAG_INFL=1e-5

curr_dir=$(pwd)
my_dir=$(dirname -- "$( readlink -f -- "$0"; )")

cd $my_dir
source ../.venv/bin/activate

./run_test_matrix $PROBLEM -n=$N_ITER lr=$LR n_steps=$N_STEPS dim=$DIM         \
    algorithm=$ALGORITHMS n_particles=$N_PARTICLES inverse_temp=$INVERSE_TEMP  \
    test_kernel=$TEST_KERNEL test_kernel_bandwidth=$TEST_KERNEL_BANDWIDTH      \
    gradient_decay=$GRADIENT_DECAY kernel_diag_infl=$KERNEL_DIAG_INFL

cd $curr_dir