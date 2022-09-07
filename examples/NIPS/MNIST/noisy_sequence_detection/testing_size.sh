#!/bin/bash

for s in $(ls -d ${1:-scenario*})
do
	echo "$s"
	for n in $(ls -d $s/noise*)
	do
		d="$n/init_digit_test_data.txt"
		echo "$d"
		cat "$d" | wc -l
	done
done
