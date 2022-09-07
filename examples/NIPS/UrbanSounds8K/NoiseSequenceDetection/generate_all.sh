#!/bin/bash

export PYTHONPATH="/home/roigvilamalam/projects/deepproblog/"

for s in $(ls -d ${1:-scenario*})
do
	echo $s
	cd $s
	python generate_data.py 9 10
	cd - > /dev/null
done
