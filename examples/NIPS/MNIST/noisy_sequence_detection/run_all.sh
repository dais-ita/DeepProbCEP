#!/bin/bash

if [ "$#" -ne 1 ]
then
          echo "Incorrect number of parameters"
            exit 1
fi

export PYTHONPATH="$PYTHONPATH:/home/roigvilamalam/projects/deepproblog/"

for size in 100 500 1000 2500 5000
# for size in 500 400 300 200 100 75 50
# for size in 5000 2500 1500 1000 750
do
	echo "Running size $size"
	python run.py --directory over_scenarios100 --scenario "_$size\$" > "outputs/$1/execution_$size.txt"
done
