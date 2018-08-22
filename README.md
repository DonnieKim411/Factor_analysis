# Factor_analysis_pkg

This is the python package performing factor analysis with Expectation Maximization method. 

This package is only tested in Linux environment and with Python 3.6.5

To download the package wheel, download [Factor_analysis_pkg_0.0.1-py3-none-any.whl](https://github.com/kdk411/Factor_analysis/blob/master/dist/Factor_analysis_pkg-0.0.1-py3-none-any.whl) under **dist** directory in this git page (or click the link)

Assuming you have activated an isolated environment (like virtualenv), run:

```
pip install Factor_analysis_pkg_0.0.1-py3-none-any.whl
```

To use the factor analysis function, prompt python terminal (or ipython) and type,

```python
from Factor_analysis_pkg import factor_analysis
```

factor_analysis function is described below:

**factor_analysis**
* **Input** 
	* **x**: A float numpy array. This is the observed variables for each data points 
	* **nFactors**: Integer. Number of latent variables to be considered

* **Output**
	* **W**: A float numpy array. A factor loading matrix
	* **z**: A float numpy array. A latent variable matrix
	* **log_history**: A list. A record of log likelihood over iterations

Note that the output **z** is mostly the main concern as the factor analysis is often used for *feature dimensionality reduction*

The package also includes an illustrative example of factor analysis using IRIS dataset for feature dimensionality reduction:

```python
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