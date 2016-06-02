#!/bin/bash

for i in $( seq 5 9 ); do
   ./build/test-decision-trees ../../logreg/data/mnist_data_training.$i.mtx ../../logreg/data/mnist_lbl_training_1.$i.mtx
   sleep 10;
done
