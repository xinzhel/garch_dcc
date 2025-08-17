import warnings
warnings.filterwarnings("ignore")
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
import numpy as np
from scipy.optimize import minimize

def dcc_loglikelihood(params, eps):
    """
    params: array-like, [alpha, beta]
    eps:    standardized residuals
    Returns negative log-likelihood for minimization.
    """
    alpha, beta = params  # Unpack DCC parameters
    T, n = eps.shape      # Number of observations and series
    # Compute the unconditional correlation matrix R_bar
    R_bar = (eps.T @ eps) / T  
    
    # Initialize Q_{t-1} with R_bar
    Q_prev = R_bar.copy()  # Q_{0} = bar{R}
    
    total_ll = 0.0  # Initialize sum of log-likelihood contributions
    
    # Loop over time points starting from t=1
    for t in range(1, T):
        # Update Q_t according to DCC recursion
        outer_eps = np.outer(eps[t-1], eps[t-1])  # ν_{t-1} ν_{t-1}'
        Q_t = ((1 - alpha - beta) * R_bar +
               alpha * outer_eps +
               beta * Q_prev)
        
        # R_t
        diag_sqrt = np.sqrt(np.diag(Q_t))           # sqrt of diagonal elements
        R_t = Q_t / np.outer(diag_sqrt, diag_sqrt)  # R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)
        R_t = np.clip(R_t, -0.9999, 0.9999) # Ensure numerical stability
        # Composite likelihood: sum of bivariate normal log-likelihoods
        for i in range(n):
            for j in range(i + 1, n):
                e_i = eps[t, i]
                e_j = eps[t, j]
                # (TODO: Check the correctness of the log-likelihood formula (Xinzhe))
                ll_ij = -0.5 * (np.log(1 - R_t[i, j]**2) +
                                (e_i**2 - 2 * R_t[i, j] * e_i * e_j + e_j**2) / (1 - R_t[i, j]**2))
                # R_t[i, j]: conditional correlation at (i,j) for time t
                total_ll += ll_ij
        
        # Update Q_prev for next iteration
        Q_prev = Q_t
    
    return -total_ll  # Negative for minimization


def fit_dcc(eps):
    """Estimate α and β by maximizing composite likelihood"""
    initial = np.array([0.01, 0.98])
    bounds = [(0, 1), (0, 1)]
    constr = {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]}
    result = minimize(dcc_loglikelihood, initial, args=(eps,),
                      bounds=bounds, constraints=constr, method='SLSQP')
    if not result.success:
        raise RuntimeError("DCC optimization failed: " + result.message)
    return result.x

def dcc_estimation(eps, alpha_hat, beta_hat):
    T, n = eps.shape
    alpha, beta = alpha_hat, beta_hat
    # 1. Compute the unconditional correlation matrix R_bar
    R_bar = np.corrcoef(eps, rowvar=False)

    # 2. Allocate array to store Q_t for each t
    Q_ts = np.zeros((T, n, n))

    # 3. Initialize at t = 0: Q_0 = R_bar
    Q_ts[0] = R_bar.copy()
    Q_prev = Q_ts[0]

    # 4. Loop over t = 1, 2, ..., T-1 to update Q_t
    for t in range(1, T):
        # 4a. Outer product of previous standardized residuals
        outer_eps = np.outer(eps[t-1], eps[t-1])
        
        # 4b. DCC recursion: Q_t = (1 - α - β) R_bar + α (ε_{t-1} ε_{t-1}ᵀ) + β Q_{t-1}
        Q_t = (1 - alpha - beta) * R_bar + alpha * outer_eps + beta * Q_prev
        
        # 4c. Store and update
        Q_ts[t] = Q_t
        Q_prev = Q_t

    return Q_ts

# load data
file_name = f'tweets_southport_3hcount_31lbl.json'
with open(file_name, 'r') as f:
    all_series = json.load(f)
df = pd.DataFrame(all_series).set_index('date')
df.index = pd.to_datetime(df.index).to_numpy(dtype='datetime64[ns]')

# slice df by date
start = np.datetime64('2024-08-03T00:00:00')
end = np.datetime64('2024-08-10T15:59:59')  # effectively represents "latest possible date"
df = df[(df.index >= start) & (df.index <= end)]

# Garch Estimation
std_resids = pd.DataFrame(index=df.index, columns=df.columns)
conditional_volatility = pd.DataFrame(index=df.index, columns=df.columns)
for col in df.columns:
    am = arch_model(df[col], mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp='off')
    mu = res.params['mu']
    conditional_volatility[col] = res.conditional_volatility
    std_resids[col] = (list(df[col].astype(float)) - mu) / conditional_volatility[col]
    
    
eps = std_resids.values
alpha_hat, beta_hat = fit_dcc(eps)
print(f"DCC parameters: alpha = {alpha_hat}, beta = {beta_hat}")

Q_ts = dcc_estimation(eps, alpha_hat, beta_hat)

# save Q_ts to a file 
np.save("Q_ts.npy", Q_ts)
print(f"Q_ts (shape: {Q_ts.shape}) saved to Q_ts.npy")

# save series names to a file
with open("series_names.txt", "w") as f:
    for name in df.columns:
        f.write(f"{name}\n")


