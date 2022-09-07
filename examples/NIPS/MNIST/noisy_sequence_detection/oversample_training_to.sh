#!/bin/bash

for s in $(ls -d $1)
do
	for n in $(ls -d $s/noise*)
	do
		d="$n/init_train_data.txt"
		echo "$d"
		#head -n $2 $d > "${d}_"
		if [[ "$d" =~ .*"scenario108".* ]];
		then
		  python oversample_108.py ${d} $2 "${d}_" > /dev/null
		else
		  python oversample.py ${d} $2 "${d}_" > /dev/null
    fi
		mv "${d}_" "$d"
	done
done
