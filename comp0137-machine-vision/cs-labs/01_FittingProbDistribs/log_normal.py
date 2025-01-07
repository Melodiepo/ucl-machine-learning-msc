import numpy as np

def log_normal(X, mu, sigma):
    """Return log-likelihood of data given parameters"

    Computes the log-likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar log-likelihood
    """
    # Edge case: sigma is zero or negative
    if sigma <= 0:
        return -np.inf  # Log-likelihood is undefined for sigma <= 0
    
    n = len(X)  # number of data points
    
    # First term: -N/2 * log(2 * pi * sigma^2)
    first_term = -n / 2 * np.log(2 * np.pi * sigma ** 2)
    
    # Second term: -1/(2 * sigma^2) * sum((X - mu)^2)
    second_term = -np.sum((X - mu) ** 2) / (2 * sigma ** 2)
    
    # Calculate log-likelihood
    loglik = first_term + second_term
    
    return loglik