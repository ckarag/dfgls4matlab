# Dickey-Fuller GLS test for MATLAB

The MATLAB DFGLS function implements unit-root testing using the DF-GLS test of Elliott, Rothenberg & Stock (1996), allowing for 3 methods for choosing the optimal lag-length in the underlying Augmented Dickey-Fuller (ADF) regression. The methods for selecting the optimal number of lags are: 
* SIC, 
* MAIC by Ng-Perron (2001), and the 
* Sequential t-test by Ng-Perron (1995).
 
For a quick tutorial and a comparison to Stata®'s dfgls function, see the Examples section in File Exchange, or the included .mlx file.
