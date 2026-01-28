#!/bin/sh

#SBATCH -p mit_normal_gpu
#SBATCH -c 32
#SBATCH -G 1

LR=0.5
GRAD_ALDI_LR=1e-2
GRADFREE_ALDI_LR=1e-2
N_STEPS=100
DIM=64
PROBLEM=aristoff_bangerth
ALGORITHMS=msip_fredholm,msip_gradientfree,msip_gradientinformed,grad_aldi,gradfree_aldi,eks,svgd
N_ITER=10
N_PARTICLES=10,20,40,80,160
INNER_QUAD=spherical_MC_radial_Laguerre
INNER_QUAD_N_SPHERICAL=5
INVERSE_TEMP=0.9 # For CBS
TEST_KERNEL=sqexp
TEST_KERNEL_LENGTH_SCALE=64.0
GRADIENT_DECAY=0.9
KERNEL_DIAG_INFL=1e-5
BOUNDS="(-10;10)"
DEVICE=cuda

curr_dir=$(pwd)
my_dir=$(dirname -- "$( readlink -f -- "$0"; )")

cd $my_dir
source ../.venv/bin/activate

./run_test_matrix $PROBLEM -n=$N_ITER -G lr=$LR n_steps=$N_STEPS dim=$DIM        \
    algorithm=$ALGORITHMS n_particles=$N_PARTICLES inverse_temp=$INVERSE_TEMP    \
    test_kernel=$TEST_KERNEL test_kernel_length_scale=$TEST_KERNEL_LENGTH_SCALE  \
    gradient_decay=$GRADIENT_DECAY kernel_diag_infl=$KERNEL_DIAG_INFL            \
    bounds=$BOUNDS grad_aldi_lr=$GRAD_ALDI_LR gradfree_aldi_lr=$GRADFREE_ALDI_LR \
    inner_quad=$INNER_QUAD inner_quad_N_spherical=$INNER_QUAD_N_SPHERICAL device=$DEVICE

cd $curr_dir