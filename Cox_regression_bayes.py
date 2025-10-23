import arviz as az
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel, SelectKBest

# Set seeds and style
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-white")

def make_exposure_matrix(time, event, breaks):
    """
    Convert individual survival data into "piecewise interval" format
    for the Bayesian proportional hazards model.

    Each subject's observation time is divided across several
    time intervals (defined by `breaks`), and for each subject-interval pair,
    we record:
      - how long the subject was at risk in that interval ("exposure")
      - whether the event (e.g., death) occurred in that interval ("conv")

    This structure lets us model survival data as Poisson counts, where
    the event count per interval follows:
        events[i,j] ~ Poisson(exposure[i,j] * hazard[i,j])

    Parameters
    ----------
    time : array-like, shape (n_subjects,)
        The observed survival or censoring time for each subject.
        (e.g., 12.3 months until death or censoring)

    event : array-like, shape (n_subjects,)
        Event indicator: 1 if event occurred, 0 if right-censored.

    breaks : array-like, shape (n_intervals + 1,)
        Cut points that define the time intervals (bins) over which
        the baseline hazard is assumed constant.
        Example: [0, 3, 6, 9, 12] → 4 intervals of 3 months each.

    Returns
    -------
    exposure : ndarray, shape (n_subjects, n_intervals)
        Amount of time each subject contributed ("was at risk")
        in each time interval. Units are the same as `time`.

    conv : ndarray, shape (n_subjects, n_intervals)
        Event indicator per interval: 1 if the event occurred in that
        interval for that subject, otherwise 0.
    """

    # Number of subjects (rows)
    n = len(time)

    # Number of time intervals (columns)
    # If breaks = [0, 3, 6, 9, 12], then m = 4
    m = len(breaks) - 1

    # Initialize matrices of zeros
    exposure = np.zeros((n, m))  # exposure time per subject-interval
    conv = np.zeros((n, m))      # event indicator (0/1) per subject-interval

    # Loop over each subject i
    for i in range(n):

        # Loop over each time interval j
        for j in range(m):
            start, end = breaks[j], breaks[j + 1]

            # Compute how much time subject i spent in interval j
            # If subject's total time < start, they contributed nothing
            # If subject's total time > end, they were at risk for the full interval
            # Otherwise, they were at risk from start until their event/censor time.
            exposure[i, j] = np.clip(min(time[i], end) - start, 0, None)

            # If the event occurred within this interval, mark it as 1
            # event[i] == 1 → subject experienced event
            # time[i] <= end → event happened before end of this interval
            # time[i] > start → event happened after interval start
            # Together → event occurred inside (start, end]
            if event[i] == 1 and time[i] <= end and time[i] > start:
                conv[i, j] = 1

    # Return both matrices
    return exposure, conv


def cox_bayesian_model(df, n_intervals=20, method="Ridge",ridge_sigma=1.0,
    lasso_b=0.5, # b for Laplace prior (lasso)
    horseshoe_tau_scale=5.0, plot_dist_sample=False):    

    '''Function to run cox proportional hazard model but using bayesian stats, df contains features, followed by 2 columns event and time, event is a 0,1 column'''
    n_patients = df.shape[0]
    patients = np.arange(n_patients)
    breaks = np.linspace(0, df["time"].max(), n_intervals + 1)
    if plot_dist_sample==True:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(
            df[df.event == 0].time.values,
            bins=breaks,
            lw=0,
            color="navy",
            alpha=0.5,
            label="CTRL",
        )
        ax.hist(
            df[df.event == 1].time.values,
            bins=breaks,
            lw=0,
            color="green",
            alpha=0.5,
            label="PD",
        )
        ax.set_xlim(0, breaks[-1])
        ax.set_xlabel("Years from sample collection")
        ax.set_yticks(range(15))
        ax.set_ylabel("Number of observations")
        ax.legend()
        plt.show()

        # -----------------------------------------------------------
    # 1. PREPARE DATA
    # -----------------------------------------------------------

    # Standardize covariates
    print("Scaling the data with standard scaler")
    sc = StandardScaler()
    X = sc.fit_transform(df.drop(["time", "event"], axis=1))
    n, p = X.shape

    # Build exposure and event matrices

    exposure, conv = make_exposure_matrix(df["time"], df["event"], breaks)

    # -----------------------------------------------------------
    # 2. SANITY CHECKS
    # -----------------------------------------------------------
    print("Shapes -> X:", X.shape, "| exposure:", exposure.shape, "| conv:", conv.shape)
    print("Unique conv values:", np.unique(conv))
    print("Exposure min/max:", exposure.min(), exposure.max())

    # -----------------------------------------------------------
    # 3. BUILD AND RUN MODEL
    # -----------------------------------------------------------
    with pm.Model() as model:
        # ------------------------------------------------------------
        # 1. PRIORS (our beliefs before seeing the data)
        # ------------------------------------------------------------
        # lambda0: baseline hazard rate for each time interval
        #   - One value per time interval (shape = n_intervals)
        #   - Modeled with a Gamma distribution so it’s always positive
        #   - A small alpha/beta (0.01, 0.01) gives a weakly-informative prior
        lambda0 = pm.Gamma("lambda0", alpha=0.01, beta=0.01, shape=n_intervals)

        # ---------- Shrinkage prior selection ----------
        if method.lower() == "ridge":
            # Normal(0, sigma) => ridge-like shrinkage
            beta = pm.Normal("beta", mu=0.0, sigma=ridge_sigma, shape=p)

        elif method.lower() == "lasso":
            # Laplace(0, b) => lasso-like (L1) shrinkage
            # Note: Laplace is supported by PyMC
            beta = pm.Laplace("beta", mu=0.0, b=lasso_b, shape=p)

        elif method.lower() == "horseshoe":
            # Horseshoe prior (hierarchical shrinkage)
            # Global scale
            tau = pm.HalfCauchy("tau", beta=horseshoe_tau_scale)
            # Local scales (one per coefficient)
            lam = pm.HalfCauchy("lam", beta=1.0, shape=p)
            # Unscaled coefficients
            beta_tilde = pm.Normal("beta_tilde", mu=0.0, sigma=1.0, shape=p)
            # Construct final beta and expose deterministically
            beta = pm.Deterministic("beta", beta_tilde * lam * tau)

        else:
            raise ValueError("method must be one of 'ridge', 'lasso', 'horseshoe'")

        # ------------------------------------------------------------
        # 2. LINEAR PREDICTOR AND HAZARD
        # ------------------------------------------------------------
        # linpred: the linear combination of predictors for each subject
        #   linpred_i = X_i ⋅ beta  (like in logistic or linear regression)
        #   It represents each subject’s risk level given their covariates.
        linpred = pt.dot(X, beta)

        # lambda_: subject- and interval-specific hazard
        #   We combine:
        #     - baseline hazard (lambda0 for each interval)
        #     - individual relative risk (exp(linpred))
        #   Broadcasting with [:, None] and [None, :] creates a full matrix:
        #     lambda_[i, j] = exp(linpred[i]) * lambda0[j]
        lambda_ = pm.Deterministic("lambda_", pt.exp(linpred)[:, None] * lambda0[None, :])

        # ------------------------------------------------------------
        # 3. EXPECTED NUMBER OF EVENTS (Poisson mean)
        # ------------------------------------------------------------
        # mu: expected number of events per subject per interval
        #   mu[i, j] = exposure[i, j] * lambda_[i, j]
        #   - exposure[i, j] = how long subject i was “at risk” in interval j
        #   - lambda_[i, j] = their hazard in that interval
        #   Adding 1e-6 avoids problems if exposure = 0.
        mu = pm.Deterministic("mu", exposure * lambda_)

        # ------------------------------------------------------------
        # 4. OBSERVED DATA (the likelihood)
        # ------------------------------------------------------------
        # conv: a 0/1 matrix showing whether the event happened in each interval
        #   - We flatten both conv and mu to 1D vectors so they match in shape
        #   - Each element represents a subject–interval observation
        #   - The number of observed events follows a Poisson distribution
        #     with mean = expected events (mu)
        obs = pm.Poisson("obs", mu, observed=conv)

        # ------------------------------------------------------------
        # 5. SAMPLING (estimate posterior distributions)
        # ------------------------------------------------------------
        # Draw samples from the posterior using MCMC (NUTS sampler)
        #   - 1000 draws after 1000 tuning steps - because of the tune we no longer need to skip burn in
        #   - target_accept=0.9 improves stability (fewer divergences) - there was a warning asking to do that
        #   - cores=12 runs chains in parallel for speed
        trace = pm.sample(1000, tune=1000, cores=12, random_seed=RANDOM_SEED, target_accept=0.9)

    # -----------------------------------------------------------
    # 4. POSTERIOR SUMMARY
    # -----------------------------------------------------------
    print(pm.summary(trace, var_names=["beta", "lambda0"]))
    pm.plot_trace(trace, var_names=["beta"])
    plt.show()

# Creating summary plots and dataframes
    feature_names = df.drop(["time", "event"], axis=1).columns
    # Get the summary dataframe
    beta_summary = az.summary(trace, var_names=["beta"], round_to=3)
    #  Add feature names to the beta summary index
    beta_summary.index = feature_names
    # Print a nice labeled summary
    print("Posterior summary for beta (covariate effects):")
    # Rank by mean effect
    beta_summary["abs_mean"] = beta_summary["mean"].abs()
    important_features = beta_summary.sort_values("abs_mean", ascending=False)
    display(important_features[["mean", "sd", "hdi_3%", "hdi_97%", "abs_mean"]])
    beta_summary["significant"] = ~(
        (beta_summary["hdi_3%"] < 0) & (beta_summary["hdi_97%"] > 0)
    )
    significant_features = beta_summary[beta_summary["significant"]]
    print("Significant (credible) predictors:\n")
    print(significant_features)
    beta_summary=beta_summary.sort_values("mean",key=lambda x: x.abs(), ascending=False).head(20)
    beta_summary=beta_summary.sort_values("abs_mean")
    plt.figure(figsize=(8, 10))
    plt.errorbar(
        beta_summary["mean"],
        beta_summary.index,
        xerr=[
            beta_summary["mean"] - beta_summary["hdi_3%"],
            beta_summary["hdi_97%"] - beta_summary["mean"]
        ],
        fmt="o",
        capsize=4,
    )
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Posterior mean of beta (effect on log hazard)")
    plt.ylabel("Feature")
    plt.title("Posterior estimates of covariate effects (with 94% HDI)")
    plt.xlim(beta_summary["mean"].min()-beta_summary["sd"].max() - 1.5, beta_summary["mean"].max()+beta_summary["sd"].max() + 1.5)
    plt.tight_layout()
    plt.show()


def feature_selection(X, y, method="from_model", k=50):
    """
    Perform feature selection using one of three methods:
    - 'from_model': LinearSVC (L1) feature selection
    - 'select_k_best': ANOVA F-test
    - 'sfs': Sequential Feature Selector with RandomForestClassifier
    """
    if method == "from_model":
        lsvc = LinearSVC(C=1, penalty="l1", dual=False, max_iter=5000).fit(X, y) #May need to adjust C for this model, low Cs result in zero features
        model = SelectFromModel(lsvc, prefit=True)
        selected_features = X.columns[model.get_support()]
        X_new = pd.DataFrame(model.transform(X), columns=selected_features, index=X.index)

    elif method == "select_k_best":
        skb = SelectKBest(f_classif, k=min(k, X.shape[1]))
        X_new_array = skb.fit_transform(X, y)
        selected_features = X.columns[skb.get_support()]
        X_new = pd.DataFrame(X_new_array, columns=selected_features, index=X.index)

    elif method == "sfs":
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        sfs = SequentialFeatureSelector(rf, n_features_to_select=min(k, X.shape[1]), n_jobs=-1)
        sfs.fit(X, y)
        selected_features = X.columns[sfs.get_support()]
        X_new = pd.DataFrame(sfs.transform(X), columns=selected_features, index=X.index)

    else:
        raise ValueError(f"Invalid method '{method}'. Choose from ['from_model', 'select_k_best', 'sfs'].")

    return X_new
