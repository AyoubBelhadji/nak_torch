#!/bin/sh

LR=1e-2
N_STEPS=100
DIM=2
PROBLEM=joker
ALGORITHMS=msip_fredholm
N_ITER=2
N_PARTICLES=10
INVERSE_TEMP=0.9 # For CBS
TEST_KERNEL=sqexp
TEST_KERNEL_LENGTH_SCALE=1.
GRADIENT_DECAY=0.9
KERNEL_DIAG_INFL=1e-3
BOUNDS="(-10;10)"
KERNEL_LENGTH_SCALE=0.3

curr_dir=$(pwd)
my_dir=$(dirname -- "$( readlink -f -- "$0"; )")

cd $my_dir
source ../.venv/bin/activate

./run_test_matrix $PROBLEM -n=$N_ITER lr=$LR n_steps=$N_STEPS dim=$DIM         \
    algorithm=$ALGORITHMS n_particles=$N_PARTICLES inverse_temp=$INVERSE_TEMP  \
    test_kernel=$TEST_KERNEL test_kernel_length_scale=$TEST_KERNEL_LENGTH_SCALE\
    gradient_decay=$GRADIENT_DECAY kernel_diag_infl=$KERNEL_DIAG_INFL          \
    bounds=$BOUNDS kernel_length_scale=$KERNEL_LENGTH_SCALE

cd $curr_dir