The result of DCC is saved in `Q_ts.npy`. You can load it with corresponding series names (See the presented code at the bottom). Or you can re-run garch_dcc.py or implement your own version using the snippet below.


## Required Python Packages
For Garch-DCC
```
arch==7.2.0
scipy
```

For data processing
``
pandas
numpy
matplotlib
```

## Load Data
```python
file_name = f'tweets_southport_3hcount_31lbl.json'
with open(file_name, 'r') as f:
    all_series = json.load(f)
df = pd.DataFrame(all_series).set_index('date')
df.index = pd.to_datetime(df.index).to_numpy(dtype='datetime64[ns]')
```
* The json file contains date series, the 32 series for UK and 32 series for AU. 
* With the above code, date becomes the index, and each UK/AU series become a column in a table. 
* Below are the column names for UK series. For AU series, just replace "UK" with "AU". 

UK, 
UK_REJECT_CLASHES_WITH_COUNTERPROTESTERS, 
UK_NEUTRAL_CLASHES_WITH_COUNTERPROTESTERS, 
UK_SUPPORT_CLASHES_WITH_COUNTERPROTESTERS, 
UK_REJECT_CLASHES_WITH_POLICE, 
UK_NEUTRAL_CLASHES_WITH_POLICE, 
UK_SUPPORT_CLASHES_WITH_POLICE, 
UK_NEUTRAL_GENERIC_PROTEST, 
UK_REJECT_GENERIC_PROTEST, 
UK_SUPPORT_GENERIC_PROTEST, 
UK_NEUTRAL_PHYSICAL_ASSAULTS_AND_HATE_CRIMES, 
UK_REJECT_PHYSICAL_ASSAULTS_AND_HATE_CRIMES, 
UK_SUPPORT_PHYSICAL_ASSAULTS_AND_HATE_CRIMES, 
UK_NEUTRAL_RIOTS, 
UK_REJECT_RIOTS, 
UK_SUPPORT_RIOTS, 
UK_NEUTRAL_VANDALISM_AND_ARSON, 
UK_REJECT_VANDALISM_AND_ARSON, 
UK_SUPPORT_VANDALISM_AND_ARSON, 
UK_NEUTRAL_WEAPONS, 
UK_REJECT_WEAPONS, 
UK_SUPPORT_WEAPONS, 
UK_NEUTRAL_antiracism, 
UK_REJECT_antiracism, 
UK_SUPPORT_antiracism, 
UK_NEUTRAL_OTHER, 
UK_REJECT_OTHER, 
UK_SUPPORT_OTHER, 
UK_antiforeigner, 
UK_antipolitics, 
UK_falseclaim, 
UK_mobilisation

## Garch
```python
std_resids = pd.DataFrame(index=df.index, columns=df.columns)
conditional_volatility = pd.DataFrame(index=df.index, columns=df.columns)
for col in df.columns:
    # 1. Specify & fit a GARCH(1,1) on series col
    am = arch_model(df[col], mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp='off')
    
    # 2.2 Extract estimated unconditional mean μ
    mu = res.params['mu']
    
    # 2.3 Extract conditional volatilities σ_t
    conditional_volatility[col] = res.conditional_volatility
    
    # 2.4 Compute standardized residuals ν_t = (r_t - μ) / σ_t
    std_resids[col] = (list(df[col].astype(float)) - mu) / conditional_volatility[col]
```
This will generate two variables useful for analysis:
* `std_resids` (Shape: Series Length): standardized residuals 
* `conditional_volatility` (Shape: Series Length): conditional volatility

## DCC
Since I did not find a well-established open-source implementation, I code composite likelihood (bivariate normal log-likelihood) with the following key formula

* $Q_t$ (Source from [the link from Andrea](https://vlab.stern.nyu.edu/docs/correlation/GARCH-DCC)): 
$$Q_t=\bar{R}+\alpha\left(\nu_{t-1} \nu_{t-1}^{\prime}-\bar{R}\right)+\beta\left(Q_{t-1}-\bar{R}\right)$$
* $R_t$ (Source from [Formula 4.1 in the book](https://bookdown.org/jarneric/financial_econometrics/4.1-dynamic-conditional-correlation.html?utm_source=chatgpt.com)): 

$$R_t=\operatorname{diag}\left(Q_t\right)^{-1 / 2} Q_t \operatorname{diag}\left(Q_t\right)^{-1 / 2}$$

* Pairwise Gaussian log-likelihood with $z=\left(e_i, e_j\right)^{\top}$ 

$$\ell\left(e_i, e_j \mid \rho\right)=-\frac{1}{2}\left(2 \log (2 \pi)+\log \left(1-\rho^2\right)+\frac{e_i^2-2 \rho e_i e_j+e_j^2}{1-\rho^2}\right)$$
```python
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
eps = std_resids.values
alpha_hat, beta_hat = fit_dcc(eps)
```

After estimating $\alpha$ and $\beta$, the snippet below is used for estimating the DCC.
```python
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
```
The resulting `Q_t` matrix is saved to Q_ts.npy and has shape (62, 64, 64), where 62 is the number of time steps and 64 is the number of series. 

## Direct Use of Generated Results


You can easily load `Q_t` for analysis:
```python
# load Q_ts from a file
Q_ts = np.load("Q_ts.npy")
```
TO facilitate the analysis, you can load series names from the file
```python
with open("series_names.txt", "r") as f:
    series_names = f.read().splitlines()
print(f"Loaded series names: {series_names}")
```