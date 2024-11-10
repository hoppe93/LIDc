# A line-integrated density synthetic diagnostic in C++
This is a C++ rewrite of the [LineIntegratedDensity](https://github.com/hoppe93/LineIntegratedDensity)
tool, albeit with a new algorithm for the actual integration. Whereas the old
tool would use an adaptive quadrature, this tool relies on the accuracy admitted
by the finite volume method used in DREAM.

Despite not being parallelized (yet), this tool is several orders of magnitude
faster than the Python tool.
