#!/bin/bash

output_folder="${2:-cut_scenarios100}"

rm -r "${output_folder}/"*

for s in $(ls -d ${1:-scenarios100/scenario*})
do
	echo $s
	for size in 10000 5000 2500 1500 1000 750 500 400 300 200 100 75 50
	do
		scenario=$(basename ${s})
		out="${output_folder}/${scenario}_${size}"
		cp -r $s $out
		./cut_training_to.sh $out $size
	done
done
