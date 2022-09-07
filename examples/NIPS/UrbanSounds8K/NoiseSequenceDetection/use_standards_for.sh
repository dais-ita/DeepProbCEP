#!/bin/bash

standard=${2:standard_test_and_validation}

for s in $(ls -d ${1:-over_scenarios100/scenario*})
do
	echo "$s"
	scenario=$(basename $s)
	window_size=$(echo "$scenario" | cut -d '_' -f2)
	for n in $(ls -d $s/noise*)
	do
		# echo "From $standard/$scenario/noise_0_00 to $n"
		cp "$standard/scenario100_${window_size}/noise_0_00/"*val* "$n/"
		cp "$standard/scenario100_${window_size}/noise_0_00/"*test* "$n/"
	done
done
