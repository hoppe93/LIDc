# A line-integrated density synthetic diagnostic in C++
This is a C++ rewrite of the [LineIntegratedDensity](https://github.com/hoppe93/LineIntegratedDensity)
tool, albeit with a new algorithm for the actual integration. Whereas the old
tool would use an adaptive quadrature, this tool relies on the accuracy admitted
by the finite volume method used in DREAM.

Despite not being parallelized (yet), this tool is several orders of magnitude
faster than the Python tool.


## Usage
The Python code below illustrates how to use the synthetic diagnostic framework:
```python

from DREAM import DREAMOutput
import sys

# Make sure that the LIDc Python library is in your Python path:
# (uncomment if needed)
#sys.path.append('/path/to/LIDc')

import LIDc


def main():
    # This is optional. The function 'integrate_dream()'
    # below can also take a file name as input.
    do = DREAMOutput('output.h5')

    # Line-of-sight configuration
    x0 = [0.9030, 0, 0.65]
    nhat = [0, 0, -1]

    # Return line-integrated (False) or line-averaged (True) density?
    line_averaged = False

    # Calculate line-integrated density for all time points
    t, nel = LIDc.integrate_dream(do, x0=x0, nhat=nhat, line_averaged=line_averaged)

    # ...or calculate at a specific time
    _, nel = LIDc.integrate_dream(do, x0=x0, nhat=nhat, time=TIME, line_averaged=line_averaged)

    return 0


if __name__ == '__main__':
    sys.exit(main())


```
