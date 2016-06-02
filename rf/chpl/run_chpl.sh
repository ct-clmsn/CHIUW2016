#!/bin/bash

./a.out --featurefile=../../logreg/data/mnist_data_training.5.mtx --labelfile=../../logreg/data/mnist_lbl_training_1.5.mtx 
sleep 10;
./a.out --featurefile=../../logreg/data/mnist_data_training.6.mtx --labelfile=../../logreg/data/mnist_lbl_training_1.6.mtx
sleep 10;
./a.out --featurefile=../../logreg/data/mnist_data_training.7.mtx --labelfile=../../logreg/data/mnist_lbl_training_1.7.mtx 
sleep 10;
./a.out --featurefile=../../logreg/data/mnist_data_training.8.mtx --labelfile=../../logreg/data/mnist_lbl_training_1.8.mtx 
sleep 10;
./a.out --featurefile=../../logreg/data/mnist_data_training.9.mtx --labelfile=../../logreg/data/mnist_lbl_training_1.9.mtx
sleep 10;

