#!/bin/bash

if [ "$#" -ne 1 ]
then
          echo "Incorrect number of parameters"
            exit 1
fi

export PYTHONPATH="$PYTHONPATH:/home/roigvilamalam/projects/deepproblog/"

for scenario in "scenario103" "scenario104" "scenario108"
# for size in 500 400 300 200 100 75 50
# for size in 5000 2500 1500 1000 750
do
	echo "Running $scenario"
	python run.py --directory balanced_noisy_scenarios --scenario "$scenario" > "outputs/$1/execution_$scenario.txt"
done
