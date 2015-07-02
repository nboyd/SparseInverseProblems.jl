#!/usr/bin/env sh
mkdir -p data
cd data

wget http://i.stanford.edu/hazy/victor/jellyfish-with-datasets.tar.bz2
tar xvf jellyfish-with-datasets.tar.bz2
rm jellyfish-with-datasets.tar.bz2
