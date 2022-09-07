#!/bin/bash

for s in $(ls -d ${1:-scenario*})
do
	echo "$s"
	for n in $(ls -d $s/noise*)
	do
		d="$n/init_val_data.txt"
		echo "$d"
		for c in "X = Y" "sequence0 = false" "sequence1 = false" "sequence2 = false" "sequence3 = false" "sequence4 = false" "sequence0 = true" "sequence1 = true" "sequence2 = true" "sequence3 = true" "sequence4 = true"
		do
			echo "${c}"
			cat "$d" | grep "${c}" | wc -l
		done
	done
done
