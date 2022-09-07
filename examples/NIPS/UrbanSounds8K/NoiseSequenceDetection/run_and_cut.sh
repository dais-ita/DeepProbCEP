#!/bin/bash

for size in 5000 2500 1500 1000 750 500 400 300 200 100 75 50
# for size in 500 400 300 200 100 75 50
# for size in 5000 2500 1500 1000 750
do
	./cut_training_to.sh "scenario100*" $size
	python run.py --scenario '100_.\Z' --noise noise > "out_$size.txt"
done
