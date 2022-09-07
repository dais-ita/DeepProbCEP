#!/bin/bash

for s in $(ls -d ${1:-scenario*})
do
	echo "$s"
	for n in $(ls -d $s/noise*)
	do
		d="$n/init_train_data.txt"
		copy="$n/init_train_data_wildcards.txt"
		echo "$d"
		cp "$d" "$copy"
		cat "$copy" | grep -v "X = Y" > "$d"
	done
done
