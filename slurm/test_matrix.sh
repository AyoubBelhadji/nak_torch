#!/bin/sh

LR=0.1
N_STEPS=1000
DIM=2
PROBLEM=joker
ALGORITHMS=cbs,grad_aldi,svgd
N_ITER=1
N_PARTICLES=10,20,40,80,160,320,640
INVERSE_TEMP=0.9 # For CBS
TEST_KERNEL=sqexp

curr_dir=$(pwd)
my_dir=$(dirname -- "$( readlink -f -- "$0"; )")
cd $my_dir
source ../.venv/bin/activate
./run_test_matrix $PROBLEM n=$N_ITER lr=$LR n_steps=$N_STEPS dim=$DIM algorithm=$ALGORITHMS n_particles=$N_PARTICLES inverse_temp=$INVERSE_TEMP test_kernel=$TEST_KERNEL
cd $curr_dir