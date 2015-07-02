#!/usr/bin/env sh
cd c
gcc-5 -O3 -c -fPIC -fopenmp matrix_completion.c
gcc-5 -fopenmp -shared -Wl,-install_name,libmc.so.1 -o libmc.so.1.0   *.o
