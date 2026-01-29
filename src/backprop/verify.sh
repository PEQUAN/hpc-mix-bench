#!/usr/bin/env bash
set -ue

OUTPUT=1 ./backprop 65536
diff output.dat ../../results/backprop/output.dat
