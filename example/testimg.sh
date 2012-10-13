#!/bin/sh

# The script tests all the png images in data folder
echo -------------------------------
echo The results are printed in green on top of the image file
echo -------------------------------
echo Images are displayed every second
echo -------------------------------

/usr/local/bin/testimg ../data/mnist.xml ../data/*.png


