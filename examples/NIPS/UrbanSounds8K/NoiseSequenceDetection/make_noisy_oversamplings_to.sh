#!/bin/bash

output_folder="${2:-balanced_noisy_scenarios}"
size=${3:-1000}

rm -r "${output_folder}/"*

for s in $(ls -d ${1:-noisy_scenarios/scenario*})
do
	echo $s

	scenario=$(basename ${s})
  out="${output_folder}/${scenario}_${size}"
  cp -r $s $out
  ./oversample_training_to.sh $out $size
done
