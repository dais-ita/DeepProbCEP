#!/bin/bash

for s in $(ls -d ${1:-over_scenarios100/scenario*})
do
	echo "$s"
	for n in $(ls -d $s/noise*)
	do
		d="$n/init_train_data.txt"
		clean="$n/init_train_data_clean.txt"
		echo "$d"
		cat "$d" | grep -v "X = Y" > "$clean"
	done
done
