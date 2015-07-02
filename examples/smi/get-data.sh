#!/usr/bin/env sh
mkdir -p data
cd data

wget http://bigwww.epfl.ch/smlm/challenge/datasets/Bundled_Tubes_Long_Sequence/sequence.zip
wget http://www.stat.berkeley.edu/~nickboyd/groundTruth.csv
wget http://www.stat.berkeley.edu/~nickboyd/file-description.xml
wget http://bigwww.epfl.ch/smlm/tools/CompareLocalization.jar
unzip sequence.zip
rm sequence.zip
