# Function for plotting model coefficients for different alphas
def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")

def cox_regression_elastic_net_cv(Xt, y, n_splits=3, l1_ratio=0.87, n_highlight=5, alpha_min_ratio=0.1, scale=True):
    """
    Perform cross-validated Cox Elastic Net regression with optimal alpha selection.
    
    Parameters:
    -----------
    Xt : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.DataFrame or structured array
        Target with 'Status' and 'Time' columns
    n_splits : int
        Number of CV folds (default: 5)
    l1_ratio : float
        Elastic net mixing parameter (default: 0.9)
    n_highlight : int
        Number of top features to highlight in plot

    scale : boolean - if scale True then standard scaler is applied
        
    """
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # create a pipeline with a scaler and a coxnet regression model
    if scale==True:
        coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=alpha_min_ratio))
    else:
        coxnet_pipe = make_pipeline(CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=alpha_min_ratio))
        # fit the model on the data
    coxnet_pipe.fit(Xt, y)

    # Create a dataframe with coeficients and different alphas
    coefficients_elastic_net = pd.DataFrame(
     coxnet_pipe.named_steps["coxnetsurvivalanalysis"].coef_, index=Xt.columns, columns=np.round( coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_, 5)
    )

    plot_coefficients(coefficients_elastic_net, n_highlight=n_highlight)

    # Get the estimated alphas from the model
    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

    # Perform grid search with a set of alphas from the estimatuibs
    gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in map(float, estimated_alphas)]},
    cv=cv,
    n_jobs=-1).fit(Xt, y)
    
    cv_results = pd.DataFrame(gcv.cv_results_)

    alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel("concordance index")
    ax.set_xlabel("alpha")
    ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)

    best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    best_coefs = pd.DataFrame(best_model.coef_, index=Xt.columns, columns=["coefficient"])

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index

    _, ax = plt.subplots(figsize=(6, 8))
    non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
