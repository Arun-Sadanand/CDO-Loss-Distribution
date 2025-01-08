# Simulating a CDO Loss Distribution

[In this project](/CDO-Loss-Study.ipynb) I simulate a CDO (Collateralized Debt Obligation) contract consisting of 100 bonds each with the same face value. 
The one-period probability of default (implied from the CDS spread on each issuer) is considered as the intensity parameter of a Poisson process, which allows one to model ‘time to default’ for each issuer. 

Using the idea of a 1-factor Gaussian copula, it is possible to introduce correlation into the expected default times. 
I then assume two different correlation matrices – one with intentionally high correlations, and one with intentionally low correlations.

By running a Monte Carlo simulation, it is possible to numerically generate a loss distribution for the specified contract.
