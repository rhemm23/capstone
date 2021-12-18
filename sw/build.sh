#!/bin/bash

gcc -I./obj -I/filespace/s/sjain75/intel-fpga-bbb/BBB_cci_mpf/sw/include/ -I/filespace/s/sjain75/opae/include/ -L/filespace/s/sjain75/opae/lib -Wl,-rpath-link -Wl,/filespace/s/sjain75/opae/lib -Wl,-rpath -Wl,/filespace/s/sjain75/opae/lib -L/filespace/s/sjain75/opae/lib64 -Wl,-rpath-link -Wl,/filespace/s/sjain75/opae/lib64 -Wl,-rpath -Wl,/filespace/s/sjain75/opae/lib64 -c main.c   -o bin/main.o


gcc -I./obj -I/filespace/s/sjain75/intel-fpga-bbb/BBB_cci_mpf/sw/include/ -I/filespace/s/sjain75/opae/include/ -L/filespace/s/sjain75/opae/lib -Wl,-rpath-link -Wl,/filespace/s/sjain75/opae/lib -Wl,-rpath -Wl,/filespace/s/sjain75/opae/lib -L/filespace/s/sjain75/opae/lib64 -Wl,-rpath-link -Wl,/filespace/s/sjain75/opae/lib64 -Wl,-rpath -Wl,/filespace/s/sjain75/opae/lib64 -c afu.c   -o bin/afu.o

gcc -I./obj -I/filespace/s/sjain75/intel-fpga-bbb/BBB_cci_mpf/sw/include/ -I/filespace/s/sjain75/opae/include/ -L/filespace/s/sjain75/opae/lib -Wl,-rpath-link -Wl,/filespace/s/sjain75/opae/lib -Wl,-rpath -Wl,/filespace/s/sjain75/opae/lib -L/filespace/s/sjain75/opae/lib64 -Wl,-rpath-link -Wl,/filespace/s/sjain75/opae/lib64 -Wl,-rpath -Wl,/filespace/s/sjain75/opae/lib64 -c compiler.c   -o bin/compiler.o


gcc -o afu bin/main.o bin/afu.o bin/compiler.o  -z noexecstack -z relro -z now -pie -L/filespace/s/sjain75/opae/lib -Wl,-rpath-link -Wl,/filespace/s/sjain75/opae/lib -Wl,-rpath -Wl,/filespace/s/sjain75/opae/lib -L/filespace/s/sjain75/opae/lib64 -Wl,-rpath-link -Wl,/filespace/s/sjain75/opae/lib64 -Wl,-rpath -Wl,/filespace/s/sjain75/opae/lib64 -luuid -lopae-cxx-core -lMPF-cxx -lMPF -lopae-c


gcc -o afu_ase bin/main.o bin/afu.o bin/compiler.o  -z noexecstack -z relro -z now -pie -L/filespace/s/sjain75/opae/lib -Wl,-rpath-link -Wl,/filespace/s/sjain75/opae/lib -Wl,-rpath -Wl,/filespace/s/sjain75/opae/lib -L/filespace/s/sjain75/opae/lib64 -Wl,-rpath-link -Wl,/filespace/s/sjain75/opae/lib64 -Wl,-rpath -Wl,/filespace/s/sjain75/opae/lib64 -luuid -lopae-cxx-core -lMPF-cxx -lMPF -lopae-c-ase

