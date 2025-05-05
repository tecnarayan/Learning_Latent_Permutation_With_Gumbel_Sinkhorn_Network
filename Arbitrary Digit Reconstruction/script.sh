#!/bin/bash

# Define values
taus=(1)
pieces_list=(10)
digits=(1 2 3 4 5 6 7 8 9 0)

# Define log file
logfile="train_outputs.log"
echo "Saving all outputs to $logfile"
echo "" > "$logfile"  # Clear log file if it exists

# Loop through all combinations
for tau in "${taus[@]}"
do
  for pieces in "${pieces_list[@]}"
  do
    for digit in "${digits[@]}"
    do
      echo "Running with tau=$tau, pieces=$pieces, digit=$digit" | tee -a "$logfile"
      python3 train.py --tau $tau --pieces $pieces --digit $digit 2>&1 | tee -a "$logfile"
      echo -e "\n========================\n" >> "$logfile"
    done
  done
done
