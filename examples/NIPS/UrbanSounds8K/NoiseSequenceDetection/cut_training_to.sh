#!/bin/bash

for s in $(ls -d $1)
do
	for n in $(ls -d $s/noise*)
	do
		d="$n/init_train_data.txt"
		echo "$d"
		#head -n $2 $d > "${d}_"
		python balanced_reduce.py ${d} $2 "${d}_" > /dev/null
		mv "${d}_" "$d"
	done
done
