#!/bin/sh

echo ---------------------
echo 1. For full performance, make sure that you compiled RELEASE version, not DEBUG version!!
echo Expect it to run ~4 min on 1Ghz CPU
echo --------------------
echo 2. Also make sure that you downloaded MNIST test dataset, 
echo ungzipped it and put into current directory
echo --------------------

cd ../data
time /usr/local/bin/testmnist mnist.xml

