#!/bin/sh

make -j4
make py
make -C python install
cd benchs/learned_termination/
