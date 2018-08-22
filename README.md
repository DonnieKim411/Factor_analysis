# Factor_analysis_pkg

This is the python package performing factor analysis with Expectation Maximization method. 

This package is only tested in Linux environment and with Python 3.6.5

To download the package, download **Factor_analysis_pkg_0.0.1-py3-none-any.whl** under **dist** directory in this git page.

Assuming you have activated an isolated environment (like virtualenv), run:

`pip install Factor_analysis_pkg_0.0.1-py3-none-any.whl`

To use the factor analysis function, prompt python terminal (or ipython) and type,

`from Factor_analysis_pkg import factor_analysis`

factor_analysis function is described below:

**factor_analysis**
* Input 
	* x: a float numpy array. This is the observed variables for each data points 
	* nFactors: Integer. Number of latent variables to be considered

* Output
	* W: a float numpy array. A factor loading matrix
	* z: a float numpy array. A latent variable matrix
	* log_history: a list. A record of log likelihood over iterations

Note that the output **z** is mostly the main concern as the factor analysis is often used for *feature dimensionality reduction*

The package also includes an illustrative example of factor analysis using IRIS dataset for feature dimensionality reduction:

```
from Factor_analysis_pkg import example
example.run_demo()
```

It will print out the following statements on the terminal:

```
play with iris data
Assume 2 latent variables
Assume 3 latent variables
```
For *2 latent variable* case, it will produce a **2D scatter plot** and a **log likelihood over iterations**

For *3 latent variable* case, it will produce a **3D scatter plot** and a **log likelihood over iterations**

In total, you get 4 plots