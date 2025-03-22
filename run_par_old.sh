#!/bin/bash

# Make sure stack size is unlimited
ulimit -s unlimited

# 1) Startup CUDA Multi-Process Service (MPS)
mkdir -p /home/jovyan/mps
export CUDA_MPS_LOG_DIRECTORY=/home/jovyan/mps
nvidia-cuda-mps-control -d

# 2) Restrict to a specific GPU if needed
export CUDA_VISIBLE_DEVICES=0

# Usage check
if [ "$#" -gt 2 ] || [ "$#" -eq 0 ]; then
  echo "Usage: $0 <SM> <fill>"
  echo "  SM: number of streaming multiprocessors to run on. Must be 2,4,6,8,10,12,14"
  echo "  fill: 0 => run one copy; 1 => launch floor(14/SM) copies in parallel"
  # shutdown MPS
  echo quit | nvidia-cuda-mps-control
  exit 1
fi

SM=$1
FILL=0
if [ "$#" -eq 2 ]; then
  FILL=$2
fi

# --------------------------------------------------------------------
# 3) Choose a baseline (e.g. n=200 at SM=14) and scale with sqrt(SM/14)
#    => "weak scaling." The total grid points ~ n^2 is proportional to SM.
#    This means each SM gets roughly the same number of points to process.
#
#    Here, we also set Lx=Ly=n so that dx = Lx/n = 1, keeping resolution ~constant.
#    If you prefer to keep Lx fixed at 200, remove --Lx $Lx --Ly $Ly below.
# --------------------------------------------------------------------
n=$(awk -v sm="$SM" 'BEGIN {printf "%.0f", 200*sqrt(sm/14)}')
Lx=$n
Ly=$n
echo "Chosen n=$n for SM=$SM => domain Lx=$Lx, Ly=$Ly"

# --------------------------------------------------------------------
# 4) Limit MPS to effectively 'SM' multiprocessors by setting the
#    active thread percentage. (Same logic as your original script.)
# --------------------------------------------------------------------
case $SM in
  2)  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=15  ;;  # ~2 SM
  4)  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=29  ;;  # ~4 SM
  6)  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=43  ;;  # ~6 SM
  8)  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=58  ;;  # ~8 SM
  10) export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=72  ;;  # ~10 SM
  12) export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=86  ;;  # ~12 SM
  14) export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 ;;  # ~14 SM
  *)
    echo "Unknown value for SM=$SM"
    echo "Valid values are 2,4,6,8,10,12,14"
    echo quit | nvidia-cuda-mps-control
    exit 1
    ;;
esac

# --------------------------------------------------------------------
# 5) Launch your code ('./par') with the chosen grid size n.
#    If fill=1, launch multiple copies in parallel to fully occupy the GPU.
# --------------------------------------------------------------------
if [ "$FILL" -eq 0 ]; then
  # Run just one copy
  ./par --n $n --Lx $Lx --Ly $Ly
else
  # Launch floor(14/SM) copies in parallel
  COPIES=$((14 / SM))
  echo "Launching $COPIES copy/copies of par..."
  for (( i=1; i<=$COPIES; i++ )); do
    ./par --n $n --Lx $Lx --Ly $Ly &
  done
  wait
fi

# --------------------------------------------------------------------
# 6) Shutdown MPS service
# --------------------------------------------------------------------
echo quit | nvidia-cuda-mps-control
