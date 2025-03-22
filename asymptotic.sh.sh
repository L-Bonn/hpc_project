#!/bin/bash

# Exit if any command fails
set -e

# Output CSV file
OUTFILE="asymptotic_scaling_results.csv"
echo "GridSize,ElapsedTime" > $OUTFILE

# GPU setup (optional MPS, device selection)
export CUDA_VISIBLE_DEVICES=0
ulimit -s unlimited

# Grid sizes to loop over 
GRID_SIZES=( 50 100 150 200 250 300 350 400 450 500 550 600 )

# Number of iterations to average timing (tune as needed)
NITER=10

for GRID in "${GRID_SIZES[@]}"; do
    echo "Running grid size: $GRID"

    # Example run: replace ./par2 with your actual binary and arguments
    # Assume your binary prints total time in nanoseconds (ns)
    result=$(./par2 --n $GRID --Lx $GRID --Ly $GRID --niter $NITER)

    # Extract total time (example: your binary prints "Total time: X ns")
    ELAPSED_TIME=$(echo "$result" | grep "Elapsed time" | awk -F ":" '{print $2}' | xargs)

    # Save to CSV
    echo "$GRID,$ELAPSED_TIME" >> $OUTFILE
done

echo "Results saved to $OUTFILE"
