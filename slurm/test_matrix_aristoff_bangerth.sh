#!/bin/sh

LR=0.1
GRAD_ALDI_LR=1e-2
GRADFREE_ALDI_LR=1e-2
N_STEPS=10
DIM=64
PROBLEM=aristoff_bangerth
ALGORITHMS=gradfree_aldi
N_ITER=1
N_PARTICLES=100
INNER_QUAD=gauss_MC
INNER_QUAD_N_QUAD=5
INVERSE_TEMP=0.9 # For CBS
TEST_KERNEL=sqexp
TEST_KERNEL_LENGTH_SCALE=1.5
GRADIENT_DECAY=0.9
KERNEL_DIAG_INFL=1e-7
BOUNDS="(-10;10)"

curr_dir=$(pwd)
my_dir=$(dirname -- "$( readlink -f -- "$0"; )")

cd $my_dir
source ../.venv/bin/activate

./run_test_matrix $PROBLEM -n=$N_ITER -v lr=$LR n_steps=$N_STEPS dim=$DIM        \
    algorithm=$ALGORITHMS n_particles=$N_PARTICLES inverse_temp=$INVERSE_TEMP    \
    test_kernel=$TEST_KERNEL test_kernel_length_scale=$TEST_KERNEL_LENGTH_SCALE  \
    gradient_decay=$GRADIENT_DECAY kernel_diag_infl=$KERNEL_DIAG_INFL            \
    bounds=$BOUNDS grad_aldi_lr=$GRAD_ALDI_LR gradfree_aldi_lr=$GRADFREE_ALDI_LR \
    inner_quad=$INNER_QUAD inner_quad_N_quad=$INNER_QUAD_N_QUAD

cd $curr_dir