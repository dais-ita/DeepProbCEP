#!/bin/bash

for s in $(ls -d ${1:-scenario*})
do
	for n in $(ls -d $s/noise*)
	do
		d="$n/init_train_data.txt"
		echo "$d"
		cat $d | grep "initiatedAt(" | wc -l
		cat $d | grep "initiatedAtNoise(" | wc -l
	done
done
