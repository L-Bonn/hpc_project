#!/bin/bash

# A list of SM values you want to test
SM_VALUES=(2 4 6 8 10 12 14)

# Name of the output CSV file
OUTPUT_CSV="times_seq.csv"

# Write the CSV header
echo "SM,Time(s)" > "$OUTPUT_CSV"

# Loop over each SM value
for SM in "${SM_VALUES[@]}"; do
  echo "Running run_par.sh with SM=$SM..."

  # Capture the entire output of run_par.sh into a variable
  SCRIPT_OUTPUT=$(./run_par.sh "$SM" 0)

  # Search for the line containing "Elapsed time" and extract the numeric value
  # e.g. from: "Elapsed time [s]  :  3.14159"
  ELAPSED_TIME=$(echo "$SCRIPT_OUTPUT" | grep "Elapsed time" | awk -F ":" '{print $2}' | xargs)

  # Append the SM and the elapsed time to the CSV
  echo "$SM,$ELAPSED_TIME" >> "$OUTPUT_CSV"

  echo "  -> Elapsed time = $ELAPSED_TIME"
done

echo "All runs complete. Results in $OUTPUT_CSV"
