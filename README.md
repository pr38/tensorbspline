# tensorbspline
A scikit-learn compatible implementation of Cubic Spline/B-Spline as well as a Multidimensional/Tensor spline variation. 

The BSpline class generates a one-dimensional spline for each feature. Knots are generated via histogram splits. The TensorBSplines gets a row-wise Kron-product between the splines until there is only one matrix left. Use with caution, as a few features will grow exponentially.

Parameters for both BSpline, TensorBSplines

n_bin: int

> number of knots.

polynomial_degrees: int

> polynomial degrees

sparse: bool
> return sparse or dense matrix

Install
-------
>pip install git+https://github.com/pr38/tensorbspline
