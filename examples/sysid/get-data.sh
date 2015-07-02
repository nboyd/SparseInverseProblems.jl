#!/usr/bin/env sh
mkdir -p data
cd data

wget ftp://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/dryer.dat.gz
wget ftp://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/robot_arm.dat.gz

gunzip dryer.dat.gz robot_arm.dat.gz
