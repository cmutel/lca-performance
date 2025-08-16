# LCA performance testing and optimization

The fastest sparse solver that Brightway can use is [pardiso](https://panua.ch/pardiso/), wrapped in Python in the [pypardiso](https://github.com/haasad/PyPardiso) library. Pypardiso links to the [Intel MKL](https://en.wikipedia.org/wiki/Math_Kernel_Library) version of the pardiso solver from 2006.

This repository has testing scripts, results, and notes on various optimization approaches.
