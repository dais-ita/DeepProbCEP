#!/bin/bash

output_folder="${2:-over_scenarios100}"

for s in $(ls -d ${1:-scenarios100/scenario*})
do
	echo $s
	for size in 100 500 1000 2500 5000 10000 25000 50000 100000 400000
	do
		scenario=$(basename ${s})
		out="${output_folder}/${scenario}_${size}"
		cp -r $s $out
		./oversample_training_to.sh $out $size
	done
done
