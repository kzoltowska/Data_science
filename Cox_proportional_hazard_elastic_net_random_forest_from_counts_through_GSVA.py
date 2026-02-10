# This function allows running cox proportional hazard model from normalised log2 transformed count data
# Feature selection: gsva done on train and test seperately, linear model fitted on train to select number of features with lowest pval. Features are then filtered for variance and correlation
# Stratified Kfold CV is performed for the model to estimate coefficient stability (for each fold new GSVA is calculated)
# Xt - normalised log2 count data; y - structured array with status and time to conversion; meta - tabular metadata with covariates used in the linear model, 
# path to gmt file needed for GSVA and number of features to select in the linear model statistical analysis

def normalise(counts_df):
    dds = DeseqDataSet(
    counts=counts_df,
    metadata=pd.DataFrame({"subject":counts_df.index}, index=counts_df.index),
    design="~1",
    refit_cooks=True,
    inference=inference,
    )
    dds.fit_size_factors()
    counts_normed=pd.DataFrame(dds.layers["normed_counts"].T, index=dds.var_names, columns=dds.obs_names).apply(lambda x: np.log2(x+1))
    return counts_normed

def cox_ph_model(Xt,y, meta, path_gmt,feat_num=30):
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    # ==============================================================================
    # 1. TRAIN-TEST SPLIT
    # ==============================================================================
    print("\n" + "="*80)
    print("1. TRAIN-TEST SPLIT")
    print("="*80)
    
    # Extract event indicator for stratification
    event = y["Status"]
    
    # Split data 70-30
    count_train, count_test, y_train, y_test = train_test_split(
        Xt, y, 
        test_size=0.25, 
        stratify=event, 
        random_state=42
    )
    
    print(f"Training set: {len(count_train)} samples")
    print(f"Test set: {len(count_test)} samples")
    print(f"Events in training: {y_train['Status'].sum()} ({y_train['Status'].mean()*100:.1f}%)")
    print(f"Events in test: {y_test['Status'].sum()} ({y_test['Status'].mean()*100:.1f}%)")
    
    # ==============================================================================
    # 2. SELECT FEATURES AND FIT MODEL 
    # ==============================================================================
    print("\n" + "="*80)
    print("2. FINAL MODEL FEATURE SELECTION AND TRAINING")
    print("="*80)
    #display(count_train)
    # normalise the data with deseq2 
    print(count_train.shape)
    print(count_test.shape)
    count_train=normalise(count_train).T
    count_test=normalise(count_test).T
    print(count_train.shape)
    print(count_test.shape)
    # Feature selection 
    selection_pvals = []

    #running gsva seperately on train and test counts    
    es = gp.gsva(data=count_train.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
    X_train=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
    
    es = gp.gsva(data=count_test.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
    X_test=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
    X_train.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_").replace(".","_") for col in X_train.columns]
    X_test.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_").replace(".","_") for col in X_test.columns]
    print(X_train.shape, X_test.shape)
    #display(X_train)
    #display(X_test)
    
    # Build dataframe for statsmodels
    selection_df=X_train.merge(meta, right_index=True, left_index=True)
    #print(selection_df.columns.to_list())
    # perform analysis to get important features 
    for feature in X_train.columns:
        formula =  f"{feature} ~ C(Status)+C(gender)+age_sample_recalculated_scaled+Neutrophil_scaled+PCT_CHIMERAS_qcs_scaled+NK_cell_scaled+MEAN_ALIGNED_READ_LENGTH_qcs_scaled+Monocyte_scaled+disease_duration_diag_closest_recalculated_scaled"
        #print(formula)
        #try:
        model = smf.ols(formula, data=selection_df).fit()
            # Find the Status term automatically
        status_terms = [t for t in model.pvalues.index if t.startswith("C(Status)")]
        if len(status_terms) == 1:
            pval = model.pvalues[status_terms[0]]
        else:
            pval = np.nan
        #except:
         #   pval=np.nan
        #print(pval)
        selection_pvals.append({
            "Feature": feature,
            "Pvalue": pval
        })
        
    selection_df_p = pd.DataFrame(selection_pvals).dropna()
    #display(selection_df_p)
    # Select significant features
    selection_selected = selection_df_p.sort_values("Pvalue", ascending=True).head(feat_num).loc[:, "Feature"].tolist()
    #print(selection_selected)
    X_train=pd.DataFrame(X_train, columns=X_train.columns)[selection_selected]
    X_test=pd.DataFrame(X_test, columns=X_test.columns)[selection_selected]

    # select highly variable features and drop correlated ones
    from sklearn.feature_selection import VarianceThreshold
    vt = VarianceThreshold(threshold=0.01)
    X_train_selected = vt.fit_transform(X_train)
    X_test_selected=vt.transform(X_test)

    X_train_selected=pd.DataFrame(X_train_selected, columns=vt.get_feature_names_out())
    X_test_selected=pd.DataFrame(X_test_selected, columns=vt.get_feature_names_out())

    print(f"After removing low variance features: {X_train_selected.shape[1]} features")
    
    dcf = DropCorrelatedFeatures(threshold=0.7)
    X_train_selected=dcf.fit_transform(X_train_selected)
    X_test_selected = dcf.transform(X_test_selected)
    
    selected_features=X_train_selected.columns
    #print(selected_features)
    print(f"After removing correlated and low variance features: {X_train_selected.shape[1]} features")
    
    # Fit Cox model with standardization
    final_model = make_pipeline(StandardScaler(),CoxPHSurvivalAnalysis())
    print(X_train_selected.shape, y_train.shape)
    
    final_model.fit(X_train_selected, y_train)
    
    # Extract coefficients
    cox_model = final_model.named_steps['coxphsurvivalanalysis']
    coef_df = pd.DataFrame({
        'Feature': X_train_selected.columns,
        'Coefficient': cox_model.coef_,
        'Hazard_Ratio': np.exp(cox_model.coef_)
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\nModel Coefficients:")
    print(coef_df.to_string(index=False))
    
    # Plot coefficients
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if c > 0 else 'blue' for c in coef_df['Coefficient']]
    bars = ax.barh(range(len(coef_df)), coef_df['Coefficient'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels(coef_df['Feature'])
    ax.set_xlabel('Coefficient')
    ax.set_title('Cox Model Coefficients (Selected Features)')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot hazard ratios
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(coef_df)), coef_df['Hazard_Ratio'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels([f.replace('REACTOME_', '') for f in coef_df['Feature']])
    ax.set_xlabel('Hazard Ratio')
    ax.set_title('Hazard Ratios (Selected Features)')
    ax.axvline(1, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ==============================================================================
    # 3. CONCORDANCE INDEX (C-INDEX) + COEFFICIENT STABILITY
    # ==============================================================================
    print("\n" + "="*80)
    print("3. CONCORDANCE INDEX EVALUATION + COEFFICIENT STABILITY")
    print("="*80)
    
    # Training C-index
    train_risk = final_model.predict(X_train_selected)
    train_c_index_censored = concordance_index_censored(y_train['Status'], y_train['Time'], train_risk)[0]
    

    
    # Test C-index
    test_risk = final_model.predict(X_test_selected)
    test_c_index_censored = concordance_index_censored(y_test['Status'], y_test['Time'], test_risk)[0]
    c_index_ipcw = concordance_index_ipcw(y_train, y_test, test_risk)
    
    # Cross-validated C-index AND coefficient tracking
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    cv_scores = []
    cv_coefficients = []  # Store coefficients from each fold

    for i, (tr_idx, val_idx) in enumerate(cv.split(Xt, y["Status"]), 1):
        X_tr = Xt.iloc[tr_idx]
        X_val = Xt.iloc[val_idx]
        X_tr=normalise(X_tr).T
        X_val=normalise(X_val).T
        es = gp.gsva(data=X_tr.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
        X_tr=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
    
        es = gp.gsva(data=X_val.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
        X_val=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
        X_tr.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_") for col in X_tr.columns]
        X_val.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_") for col in X_val.columns]

        X_tr = X_tr[selected_features]
        X_val = X_val[selected_features]
    
        y_tr = y[tr_idx]
        y_val = y[val_idx]
    
        final_model.fit(X_tr, y_tr)
        val_risk = final_model.predict(X_val)
    
        c_index = concordance_index_censored(y_val["Status"], y_val["Time"], val_risk)[0]
    
        # Extract coefficients from this fold
        fold_coef = final_model.named_steps['coxphsurvivalanalysis'].coef_
        cv_coefficients.append(fold_coef)
        
        cv_scores.append(c_index)
        print(f"Fold {i}: C-index = {c_index:.4f}")
    
    cv_scores = np.array(cv_scores)
    cv_coefficients = np.array(cv_coefficients)  # Shape: (n_folds, n_features)
    
    print(f"CV C-index: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    print(f"\nTraining C-index: {train_c_index_censored:.4f}")
    print(f"Test C-index: {test_c_index_censored:.4f}")
    print(f"C-index IPCW: {c_index_ipcw}")
    print(f"Cross-validated C-index: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"CV scores: {cv_scores}")

    # ==============================================================================
    # 4. COEFFICIENT STABILITY ACROSS FOLDS
    # ==============================================================================
    print("\n" + "="*80)
    print("4. COEFFICIENT STABILITY ACROSS FOLDS")
    print("="*80)
    
    # Calculate mean and std of coefficients across folds
    coef_mean = cv_coefficients.mean(axis=0)
    coef_std = cv_coefficients.std(axis=0)
    
    # Create dataframe with fold-wise coefficients
    coef_stability_df = pd.DataFrame({
        'Feature': selected_features,
        'Mean_Coefficient': coef_mean,
        'Std_Coefficient': coef_std,
        'CV_Coefficient': coef_std / np.abs(coef_mean),  # Coefficient of variation
        'Mean_HR': np.exp(coef_mean),
        'Final_Model_Coef': cox_model.coef_  # From the model trained on full training set
    }).sort_values('Mean_Coefficient', key=abs, ascending=False)
    
    print("\nCoefficient Stability Across Folds:")
    print(coef_stability_df.to_string(index=False))
    
    # Plot coefficients with error bars
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by mean coefficient for better visualization
    sort_idx = np.argsort(np.abs(coef_mean))[::-1]
    sorted_features = [selected_features[i] for i in sort_idx]
    sorted_mean = coef_mean[sort_idx]
    sorted_std = coef_std[sort_idx]

    y_pos = np.arange(len(sorted_features))
    colors = ['red' if c > 0 else 'blue' for c in sorted_mean]
    
    # Plot bars with error bars
    ax.barh(y_pos, sorted_mean, xerr=sorted_std, color=colors, alpha=0.6, capsize=5, error_kw={'linewidth': 2, 'elinewidth': 2})
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('REACTOME_', '').replace('WP_', '').replace('GO_BP_', '') 
                         for f in sorted_features], fontsize=10)
    ax.set_xlabel('Coefficient', fontsize=12)
    ax.set_title('Cox Model Coefficients Across CV Folds (Mean ± SD)', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    # Plot coefficient values from each fold
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For each feature, plot the coefficient from each fold
    for i, feature_idx in enumerate(sort_idx):
        feature_name = selected_features[feature_idx]
        fold_coefs = cv_coefficients[:, feature_idx]
        
        # Plot individual fold coefficients
        ax.scatter([i]*len(fold_coefs), fold_coefs, alpha=0.6, s=100, color='red' if coef_mean[feature_idx] > 0 else 'blue')
        
        # Plot mean
        ax.scatter(i, coef_mean[feature_idx], s=200, marker='D', color='darkred' if coef_mean[feature_idx] > 0 else 'darkblue', edgecolors='black', linewidths=2, zorder=5)
    
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels([f.replace('REACTOME_', '').replace('WP_', '').replace('GO_BP_', '') for f in sorted_features], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Coefficient', fontsize=12)
    ax.set_title('Coefficient Values Across Individual CV Folds\n(Diamond = Mean, Circles = Individual Folds)', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Plot Coefficient of Variation to assess stability
    fig, ax = plt.subplots(figsize=(12, 6))
    cv_coef = coef_stability_df.sort_values('CV_Coefficient', ascending=False)
    ax.barh(range(len(cv_coef)), cv_coef['CV_Coefficient'], alpha=0.7, color='orange')
    ax.set_yticks(range(len(cv_coef)))
    ax.set_yticklabels([f.replace('REACTOME_', '').replace('WP_', '').replace('GO_BP_', '') for f in cv_coef['Feature']], fontsize=10)
    ax.set_xlabel('Coefficient of Variation (Std/|Mean|)', fontsize=12)
    ax.set_title('Coefficient Stability (Lower = More Stable)', fontsize=14, fontweight='bold')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='CV > 0.5 (Unstable)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    print("\nStability Assessment:")
    print(f"Features with CV > 0.5 (unstable): {(coef_stability_df['CV_Coefficient'] > 0.5).sum()}")
    print(f"Features with CV < 0.3 (stable): {(coef_stability_df['CV_Coefficient'] < 0.3).sum()}")
    
    # ==============================================================================
    # 5. RISK STRATIFICATION
    # ==============================================================================
    print("\n" + "="*80)
    print("5. RISK STRATIFICATION ANALYSIS")
    print("="*80)

    # calculate gsva but this time for the full data
    es = gp.gsva(data=Xt.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
    X_all=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
    X_all.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_") for col in X_all.columns]
       
    # Calculate risk scores for all data
    all_risk = final_model.predict(X_all[selected_features])
    
    # Create risk groups (tertiles)
    risk_groups = pd.qcut(all_risk, q=3, labels=['Low', 'Medium', 'High'])
    
    # Count by group
    print("\nRisk Group Distribution:")
    print(risk_groups.value_counts().sort_index())
    
    # Event rates by group
    for group in ['Low', 'Medium', 'High']:
        mask = risk_groups == group
        event_rate = y[mask]['Status'].mean()
        median_time = np.median(y[mask]['Time'])
        print(f"{group} Risk: {mask.sum()} patients, {event_rate*100:.1f}% events, median time: {median_time:.1f}")
    
    # Kaplan-Meier curves by risk group
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    for group in ['Low', 'Medium', 'High']:
        mask = risk_groups == group
        kmf.fit(y[mask]['Time'], y[mask]['Status'], label=f'{group} Risk (n={mask.sum()})')
        kmf.plot_survival_function(ax=ax, color=colors[group], linewidth=2)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Kaplan-Meier Survival Curves by Risk Group', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Log-rank test
    logrank_result = multivariate_logrank_test(y['Time'], risk_groups, y['Status'])
    print(f"\nLog-rank test p-value: {logrank_result.p_value:.4e}")
    print(f"Test statistic: {logrank_result.test_statistic:.4f}")
    
    # ==============================================================================
    # 5. UNIVARIATE ANALYSIS OF EACH FEATURE
    # ==============================================================================
    print("\n" + "="*80)
    print("5. UNIVARIATE FEATURE ANALYSIS")
    print("="*80)

    # run univariate model without cv on the initial train-test split
    univariate_results = []
    for feature in X_train_selected.columns:
        # Fit univariate model
        uni_model = make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis())
        uni_model.fit(X_train_selected[[feature]], y_train)
        
        # Get metrics
        coef = uni_model.named_steps['coxphsurvivalanalysis'].coef_[0]
        hr = np.exp(coef)
        c_index = uni_model.score(X_test_selected[[feature]], y_test)
        
        univariate_results.append({
            'Feature': feature.replace('REACTOME_', ''),
            'Coefficient': coef,
            'Hazard_Ratio': hr,
            'C_index': c_index
        })

    uni_df = pd.DataFrame(univariate_results).sort_values('C_index', ascending=False)
    print("\nUnivariate Cox Models:")
    print(uni_df.to_string(index=False))
    
    # Plot univariate C-indices
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(uni_df)), uni_df['C_index'], alpha=0.7, color='steelblue')
    ax.set_yticks(range(len(uni_df)))
    ax.set_yticklabels(uni_df['Feature'])
    ax.set_xlabel('C-index')
    ax.set_title('Univariate C-index for Each Feature')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1, label='Random')
    ax.axvline(test_c_index_censored, color='green', linestyle='--', linewidth=1, label='Multivariate Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

    # ==============================================================================
    # 6. RISK SCORE DISTRIBUTION
    # ==============================================================================
    print("\n" + "="*80)
    print("6. RISK SCORE DISTRIBUTION")
    print("="*80)
    
    # Separate by event status
    censored_risk = test_risk[y_test['Status'] == 0]
    event_risk = test_risk[y_test['Status'] == 1]
    
    print(f"\nRisk scores for censored: mean={censored_risk.mean():.3f}, std={censored_risk.std():.3f}")
    print(f"Risk scores for events: mean={event_risk.mean():.3f}, std={event_risk.std():.3f}")
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(event_risk, censored_risk)
    print(f"T-test p-value: {p_value:.4e}")
    
    # Plot distributions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(censored_risk, bins=30, alpha=0.6, label='Censored', color='blue', edgecolor='black')
    ax.hist(event_risk, bins=30, alpha=0.6, label='Event', color='red', edgecolor='black')
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Risk Score Distribution by Event Status', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ==============================================================================
    # 7. SUMMARY REPORT
    # ==============================================================================
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)

    summary = f"""
    Model Performance:
      - Training C-index: {train_c_index_censored:.4f}
      - Test C-index: {test_c_index_censored:.4f}
      - CV C-index: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
     
    
    Risk Stratification:
      - Log-rank test p-value: {logrank_result.p_value:.4e}
      - Significant separation between risk groups: {'Yes' if logrank_result.p_value < 0.05 else 'No'}
    
    Feature Summary:
      - Number of features: {len(selected_features)}
      - Protective features (HR < 1): {(coef_df['Hazard_Ratio'] < 1).sum()}
      - Risk features (HR > 1): {(coef_df['Hazard_Ratio'] > 1).sum()}
      
    Top Feature (by absolute coefficient):
      - {coef_df.iloc[0]['Feature'].replace('REACTOME_', '')}
      - Coefficient: {coef_df.iloc[0]['Coefficient']:.4f}
      - Hazard Ratio: {coef_df.iloc[0]['Hazard_Ratio']:.4f}
    """
    
    print(summary)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

# adding model using elastic net penalty
def normalise(counts_df):
    dds = DeseqDataSet(
    counts=counts_df,
    metadata=pd.DataFrame({"subject":counts_df.index}, index=counts_df.index),
    design="~1",
    refit_cooks=True,
    inference=inference,
    )
    dds.fit_size_factors()
    counts_normed=pd.DataFrame(dds.layers["normed_counts"].T, index=dds.var_names, columns=dds.obs_names).apply(lambda x: np.log2(x+1))
    return counts_normed

def cox_ph_model(Xt,y, meta, path_gmt,feat_num=30):
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    # ==============================================================================
    # 1. TRAIN-TEST SPLIT
    # ==============================================================================
    print("\n" + "="*80)
    print("1. TRAIN-TEST SPLIT")
    print("="*80)
    
    # Extract event indicator for stratification
    event = y["Status"]
    
    # Split data 70-30
    count_train, count_test, y_train, y_test = train_test_split(
        Xt, y, 
        test_size=0.25, 
        stratify=event, 
        random_state=42
    )
    
    print(f"Training set: {len(count_train)} samples")
    print(f"Test set: {len(count_test)} samples")
    print(f"Events in training: {y_train['Status'].sum()} ({y_train['Status'].mean()*100:.1f}%)")
    print(f"Events in test: {y_test['Status'].sum()} ({y_test['Status'].mean()*100:.1f}%)")
    
    # ==============================================================================
    # 2. SELECT FEATURES AND FIT MODEL 
    # ==============================================================================
    print("\n" + "="*80)
    print("2. FINAL MODEL FEATURE SELECTION AND TRAINING")
    print("="*80)
    #display(count_train)
    # normalise the data with deseq2 
    print(count_train.shape)
    print(count_test.shape)
    count_train=normalise(count_train).T
    count_test=normalise(count_test).T
    print(count_train.shape)
    print(count_test.shape)
    # Feature selection 
    selection_pvals = []

    #running gsva seperately on train and test counts    
    es = gp.gsva(data=count_train.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
    X_train=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
    
    es = gp.gsva(data=count_test.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
    X_test=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
    X_train.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_").replace(".","_") for col in X_train.columns]
    X_test.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_").replace(".","_") for col in X_test.columns]
    print(X_train.shape, X_test.shape)
    #display(X_train)
    #display(X_test)
    
    # Build dataframe for statsmodels
    selection_df=X_train.merge(meta, right_index=True, left_index=True)
    #print(selection_df.columns.to_list())
    # perform analysis to get important features 
    for feature in X_train.columns:
        formula =  f"{feature} ~ C(Status)+C(gender)+age_sample_recalculated_scaled+Neutrophil_scaled+PCT_CHIMERAS_qcs_scaled+NK_cell_scaled+MEAN_ALIGNED_READ_LENGTH_qcs_scaled+Monocyte_scaled+disease_duration_diag_closest_recalculated_scaled"
        #print(formula)
        #try:
        model = smf.ols(formula, data=selection_df).fit()
            # Find the Status term automatically
        status_terms = [t for t in model.pvalues.index if t.startswith("C(Status)")]
        if len(status_terms) == 1:
            pval = model.pvalues[status_terms[0]]
        else:
            pval = np.nan
        #except:
         #   pval=np.nan
        #print(pval)
        selection_pvals.append({
            "Feature": feature,
            "Pvalue": pval
        })
        
    selection_df_p = pd.DataFrame(selection_pvals).dropna()
    #display(selection_df_p)
    # Select significant features
    selection_selected = selection_df_p.sort_values("Pvalue", ascending=True).head(feat_num).loc[:, "Feature"].tolist()
    #print(selection_selected)
    X_train=pd.DataFrame(X_train, columns=X_train.columns)[selection_selected]
    X_test=pd.DataFrame(X_test, columns=X_test.columns)[selection_selected]

    # select highly variable features and drop correlated ones
    from sklearn.feature_selection import VarianceThreshold
    vt = VarianceThreshold(threshold=0.01)
    X_train_selected = vt.fit_transform(X_train)
    X_test_selected=vt.transform(X_test)

    X_train_selected=pd.DataFrame(X_train_selected, columns=vt.get_feature_names_out())
    X_test_selected=pd.DataFrame(X_test_selected, columns=vt.get_feature_names_out())

    print(f"After removing low variance features: {X_train_selected.shape[1]} features")
    
    dcf = DropCorrelatedFeatures(threshold=0.7)
    X_train_selected=dcf.fit_transform(X_train_selected)
    X_test_selected = dcf.transform(X_test_selected)
    
    selected_features=X_train_selected.columns
    #print(selected_features)
    print(f"After removing correlated and low variance features: {X_train_selected.shape[1]} features")
    
    # Fit Cox model with standardization
    final_model = make_pipeline(StandardScaler(),CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=100))
    print(X_train_selected.shape, y_train.shape)
    final_model.fit(X_train_selected, y_train)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    event_indicator = y_train["Status"]
    estimated_alphas = final_model.named_steps["coxnetsurvivalanalysis"].alphas_

    gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in map(float, estimated_alphas)]},
    cv=cv.split(X_train_selected, event_indicator),
    n_jobs=-1,
    ).fit(X_train_selected, y_train)
    
    final_model = gcv.best_estimator_
    print(final_model)
    final_model.fit(X_train_selected, y_train)
    
    # Extract coefficients
    cox_model = final_model.named_steps['coxnetsurvivalanalysis']
    #print(cox_model.coef_)
    coef_df = pd.DataFrame({
        'Feature': X_train_selected.columns,
        'Coefficient': cox_model.coef_.flatten().tolist(),
        'Hazard_Ratio': np.exp(cox_model.coef_.flatten().tolist())
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\nModel Coefficients:")
    print(coef_df.to_string(index=False))
    
    # Plot coefficients
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if c > 0 else 'blue' for c in coef_df['Coefficient']]
    bars = ax.barh(range(len(coef_df)), coef_df['Coefficient'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels(coef_df['Feature'])
    ax.set_xlabel('Coefficient')
    ax.set_title('Cox Model Coefficients (Selected Features)')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot hazard ratios
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(coef_df)), coef_df['Hazard_Ratio'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels([f.replace('REACTOME_', '') for f in coef_df['Feature']])
    ax.set_xlabel('Hazard Ratio')
    ax.set_title('Hazard Ratios (Selected Features)')
    ax.axvline(1, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ==============================================================================
    # 3. CONCORDANCE INDEX (C-INDEX) + COEFFICIENT STABILITY
    # ==============================================================================
    print("\n" + "="*80)
    print("3. CONCORDANCE INDEX EVALUATION + COEFFICIENT STABILITY")
    print("="*80)
    
    # Training C-index
    train_risk = final_model.predict(X_train_selected)
    train_c_index_censored = concordance_index_censored(y_train['Status'], y_train['Time'], train_risk)[0]
    

    
    # Test C-index
    test_risk = final_model.predict(X_test_selected)
    test_c_index_censored = concordance_index_censored(y_test['Status'], y_test['Time'], test_risk)[0]
    c_index_ipcw = concordance_index_ipcw(y_train, y_test, test_risk)
    
    # Cross-validated C-index AND coefficient tracking
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    cv_scores = []
    cv_coefficients = []  # Store coefficients from each fold

    for i, (tr_idx, val_idx) in enumerate(cv.split(Xt, y["Status"]), 1):
        X_tr = Xt.iloc[tr_idx]
        X_val = Xt.iloc[val_idx]
        X_tr=normalise(X_tr).T
        X_val=normalise(X_val).T
        es = gp.gsva(data=X_tr.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
        X_tr=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
    
        es = gp.gsva(data=X_val.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
        X_val=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
        X_tr.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_") for col in X_tr.columns]
        X_val.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_") for col in X_val.columns]

        X_tr = X_tr[selected_features]
        X_val = X_val[selected_features]
    
        y_tr = y[tr_idx]
        y_val = y[val_idx]
    
        final_model.fit(X_tr, y_tr)
        val_risk = final_model.predict(X_val)
    
        c_index = concordance_index_censored(y_val["Status"], y_val["Time"], val_risk)[0]
    
        # Extract coefficients from this fold
        fold_coef = final_model.named_steps['coxnetsurvivalanalysis'].coef_.flatten()
        cv_coefficients.append(fold_coef)
        
        cv_scores.append(c_index)
        print(f"Fold {i}: C-index = {c_index:.4f}")
    
    cv_scores = np.array(cv_scores)
    cv_coefficients = np.array(cv_coefficients)  # Shape: (n_folds, n_features)
    
    print(f"CV C-index: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    print(f"\nTraining C-index: {train_c_index_censored:.4f}")
    print(f"Test C-index: {test_c_index_censored:.4f}")
    print(f"C-index IPCW: {c_index_ipcw}")
    print(f"Cross-validated C-index: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"CV scores: {cv_scores}")

    # ==============================================================================
    # 4. COEFFICIENT STABILITY ACROSS FOLDS
    # ==============================================================================
    print("\n" + "="*80)
    print("4. COEFFICIENT STABILITY ACROSS FOLDS")
    print("="*80)
    
    # Calculate mean and std of coefficients across folds
    coef_mean = cv_coefficients.mean(axis=0)
    coef_std = cv_coefficients.std(axis=0)
    
    # Create dataframe with fold-wise coefficients
    coef_stability_df = pd.DataFrame({
        'Feature': selected_features,
        'Mean_Coefficient': coef_mean,
        'Std_Coefficient': coef_std,
        'CV_Coefficient': coef_std / np.abs(coef_mean),  # Coefficient of variation
        'Mean_HR': np.exp(coef_mean),
        'Final_Model_Coef': cox_model.coef_.flatten()  # From the model trained on full training set
    }).sort_values('Mean_Coefficient', key=abs, ascending=False)
    
    print("\nCoefficient Stability Across Folds:")
    print(coef_stability_df.to_string(index=False))
    
    # Plot coefficients with error bars
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by mean coefficient for better visualization
    sort_idx = np.argsort(np.abs(coef_mean))[::-1]
    sorted_features = [selected_features[i] for i in sort_idx]
    sorted_mean = coef_mean[sort_idx]
    sorted_std = coef_std[sort_idx]

    y_pos = np.arange(len(sorted_features))
    colors = ['red' if c > 0 else 'blue' for c in sorted_mean]
    
    # Plot bars with error bars
    ax.barh(y_pos, sorted_mean, xerr=sorted_std, color=colors, alpha=0.6, capsize=5, error_kw={'linewidth': 2, 'elinewidth': 2})
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('REACTOME_', '').replace('WP_', '').replace('GO_BP_', '') 
                         for f in sorted_features], fontsize=10)
    ax.set_xlabel('Coefficient', fontsize=12)
    ax.set_title('Cox Model Coefficients Across CV Folds (Mean ± SD)', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    # Plot coefficient values from each fold
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For each feature, plot the coefficient from each fold
    for i, feature_idx in enumerate(sort_idx):
        feature_name = selected_features[feature_idx]
        fold_coefs = cv_coefficients[:, feature_idx]
        
        # Plot individual fold coefficients
        ax.scatter([i]*len(fold_coefs), fold_coefs, alpha=0.6, s=100, color='red' if coef_mean[feature_idx] > 0 else 'blue')
        
        # Plot mean
        ax.scatter(i, coef_mean[feature_idx], s=200, marker='D', color='darkred' if coef_mean[feature_idx] > 0 else 'darkblue', edgecolors='black', linewidths=2, zorder=5)
    
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels([f.replace('REACTOME_', '').replace('WP_', '').replace('GO_BP_', '') for f in sorted_features], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Coefficient', fontsize=12)
    ax.set_title('Coefficient Values Across Individual CV Folds\n(Diamond = Mean, Circles = Individual Folds)', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Plot Coefficient of Variation to assess stability
    fig, ax = plt.subplots(figsize=(12, 6))
    cv_coef = coef_stability_df.sort_values('CV_Coefficient', ascending=False)
    ax.barh(range(len(cv_coef)), cv_coef['CV_Coefficient'], alpha=0.7, color='orange')
    ax.set_yticks(range(len(cv_coef)))
    ax.set_yticklabels([f.replace('REACTOME_', '').replace('WP_', '').replace('GO_BP_', '') for f in cv_coef['Feature']], fontsize=10)
    ax.set_xlabel('Coefficient of Variation (Std/|Mean|)', fontsize=12)
    ax.set_title('Coefficient Stability (Lower = More Stable)', fontsize=14, fontweight='bold')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='CV > 0.5 (Unstable)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    print("\nStability Assessment:")
    print(f"Features with CV > 0.5 (unstable): {(coef_stability_df['CV_Coefficient'] > 0.5).sum()}")
    print(f"Features with CV < 0.3 (stable): {(coef_stability_df['CV_Coefficient'] < 0.3).sum()}")
    
    # ==============================================================================
    # 5. RISK STRATIFICATION
    # ==============================================================================
    print("\n" + "="*80)
    print("5. RISK STRATIFICATION ANALYSIS")
    print("="*80)

    # calculate gsva but this time for the full data
    es = gp.gsva(data=Xt.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
    X_all=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
    X_all.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_") for col in X_all.columns]
       
    # Calculate risk scores for all data
    all_risk = final_model.predict(X_all[selected_features])
    
    # Create risk groups (tertiles)
    risk_groups = pd.qcut(all_risk, q=3, labels=['Low', 'Medium', 'High'])
    
    # Count by group
    print("\nRisk Group Distribution:")
    print(risk_groups.value_counts().sort_index())
    
    # Event rates by group
    for group in ['Low', 'Medium', 'High']:
        mask = risk_groups == group
        event_rate = y[mask]['Status'].mean()
        median_time = np.median(y[mask]['Time'])
        print(f"{group} Risk: {mask.sum()} patients, {event_rate*100:.1f}% events, median time: {median_time:.1f}")
    
    # Kaplan-Meier curves by risk group
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    for group in ['Low', 'Medium', 'High']:
        mask = risk_groups == group
        kmf.fit(y[mask]['Time'], y[mask]['Status'], label=f'{group} Risk (n={mask.sum()})')
        kmf.plot_survival_function(ax=ax, color=colors[group], linewidth=2)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Kaplan-Meier Survival Curves by Risk Group', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Log-rank test
    logrank_result = multivariate_logrank_test(y['Time'], risk_groups, y['Status'])
    print(f"\nLog-rank test p-value: {logrank_result.p_value:.4e}")
    print(f"Test statistic: {logrank_result.test_statistic:.4f}")
    
    # ==============================================================================
    # 5. UNIVARIATE ANALYSIS OF EACH FEATURE
    # ==============================================================================
    print("\n" + "="*80)
    print("5. UNIVARIATE FEATURE ANALYSIS")
    print("="*80)

    # run univariate model without cv on the initial train-test split
    univariate_results = []
    for feature in X_train_selected.columns:
        # Fit univariate model
        uni_model = make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis())
        uni_model.fit(X_train_selected[[feature]], y_train)
        
        # Get metrics
        coef = uni_model.named_steps['coxphsurvivalanalysis'].coef_[0]
        hr = np.exp(coef)
        c_index = uni_model.score(X_test_selected[[feature]], y_test)
        
        univariate_results.append({
            'Feature': feature.replace('REACTOME_', ''),
            'Coefficient': coef,
            'Hazard_Ratio': hr,
            'C_index': c_index
        })

    uni_df = pd.DataFrame(univariate_results).sort_values('C_index', ascending=False)
    print("\nUnivariate Cox Models:")
    print(uni_df.to_string(index=False))
    
    # Plot univariate C-indices
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(uni_df)), uni_df['C_index'], alpha=0.7, color='steelblue')
    ax.set_yticks(range(len(uni_df)))
    ax.set_yticklabels(uni_df['Feature'])
    ax.set_xlabel('C-index')
    ax.set_title('Univariate C-index for Each Feature')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1, label='Random')
    ax.axvline(test_c_index_censored, color='green', linestyle='--', linewidth=1, label='Multivariate Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

    # ==============================================================================
    # 6. RISK SCORE DISTRIBUTION
    # ==============================================================================
    print("\n" + "="*80)
    print("6. RISK SCORE DISTRIBUTION")
    print("="*80)
    
    # Separate by event status
    censored_risk = test_risk[y_test['Status'] == 0]
    event_risk = test_risk[y_test['Status'] == 1]
    
    print(f"\nRisk scores for censored: mean={censored_risk.mean():.3f}, std={censored_risk.std():.3f}")
    print(f"Risk scores for events: mean={event_risk.mean():.3f}, std={event_risk.std():.3f}")
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(event_risk, censored_risk)
    print(f"T-test p-value: {p_value:.4e}")
    
    # Plot distributions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(censored_risk, bins=30, alpha=0.6, label='Censored', color='blue', edgecolor='black')
    ax.hist(event_risk, bins=30, alpha=0.6, label='Event', color='red', edgecolor='black')
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Risk Score Distribution by Event Status', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ==============================================================================
    # 7. SUMMARY REPORT
    # ==============================================================================
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)

    summary = f"""
    Model Performance:
      - Training C-index: {train_c_index_censored:.4f}
      - Test C-index: {test_c_index_censored:.4f}
      - CV C-index: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
     
    
    Risk Stratification:
      - Log-rank test p-value: {logrank_result.p_value:.4e}
      - Significant separation between risk groups: {'Yes' if logrank_result.p_value < 0.05 else 'No'}
    
    Feature Summary:
      - Number of features: {len(selected_features)}
      - Protective features (HR < 1): {(coef_df['Hazard_Ratio'] < 1).sum()}
      - Risk features (HR > 1): {(coef_df['Hazard_Ratio'] > 1).sum()}
      
    Top Feature (by absolute coefficient):
      - {coef_df.iloc[0]['Feature'].replace('REACTOME_', '')}
      - Coefficient: {coef_df.iloc[0]['Coefficient']:.4f}
      - Hazard Ratio: {coef_df.iloc[0]['Hazard_Ratio']:.4f}
    """
    
    print(summary)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

# model using random survival forests and gridsearchCV hyperparameter tuning

def normalise(counts_df):
    dds = DeseqDataSet(
    counts=counts_df,
    metadata=pd.DataFrame({"subject":counts_df.index}, index=counts_df.index),
    design="~1",
    refit_cooks=True,
    inference=inference,
    )
    dds.fit_size_factors()
    counts_normed=pd.DataFrame(dds.layers["normed_counts"].T, index=dds.var_names, columns=dds.obs_names).apply(lambda x: np.log2(x+1))
    return counts_normed

def random_surv_forest_model(Xt,y, meta, path_gmt,feat_num=30):
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    # ==============================================================================
    # 1. TRAIN-TEST SPLIT
    # ==============================================================================
    print("\n" + "="*80)
    print("1. TRAIN-TEST SPLIT")
    print("="*80)
    
    # Extract event indicator for stratification
    event = y["Status"]
    
    # Split data 70-30
    count_train, count_test, y_train, y_test = train_test_split(
        Xt, y, 
        test_size=0.25, 
        stratify=event, 
        random_state=42
    )
    
    print(f"Training set: {len(count_train)} samples")
    print(f"Test set: {len(count_test)} samples")
    print(f"Events in training: {y_train['Status'].sum()} ({y_train['Status'].mean()*100:.1f}%)")
    print(f"Events in test: {y_test['Status'].sum()} ({y_test['Status'].mean()*100:.1f}%)")
    
    # ==============================================================================
    # 2. SELECT FEATURES AND FIT MODEL 
    # ==============================================================================
    print("\n" + "="*80)
    print("2. FINAL MODEL FEATURE SELECTION AND TRAINING")
    print("="*80)
    #display(count_train)
    # normalise the data with deseq2 
    print(count_train.shape)
    print(count_test.shape)
    count_train=normalise(count_train).T
    count_test=normalise(count_test).T
    print(count_train.shape)
    print(count_test.shape)
    # Feature selection 
    selection_pvals = []

    #running gsva seperately on train and test counts    
    es = gp.gsva(data=count_train.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
    X_train=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
    
    es = gp.gsva(data=count_test.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
    X_test=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
    X_train.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_").replace(".","_") for col in X_train.columns]
    X_test.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_").replace(".","_") for col in X_test.columns]
    print(X_train.shape, X_test.shape)
    #display(X_train)
    #display(X_test)
    
    # Build dataframe for statsmodels
    selection_df=X_train.merge(meta, right_index=True, left_index=True)
    #print(selection_df.columns.to_list())
    # perform analysis to get important features 
    for feature in X_train.columns:
        formula =  f"{feature} ~ C(Status)+C(gender)+age_sample_recalculated_scaled+Neutrophil_scaled+PCT_CHIMERAS_qcs_scaled+NK_cell_scaled+MEAN_ALIGNED_READ_LENGTH_qcs_scaled+Monocyte_scaled+disease_duration_diag_closest_recalculated_scaled"
        #print(formula)
        #try:
        model = smf.ols(formula, data=selection_df).fit()
            # Find the Status term automatically
        status_terms = [t for t in model.pvalues.index if t.startswith("C(Status)")]
        if len(status_terms) == 1:
            pval = model.pvalues[status_terms[0]]
        else:
            pval = np.nan
        #except:
         #   pval=np.nan
        #print(pval)
        selection_pvals.append({
            "Feature": feature,
            "Pvalue": pval
        })
        
    selection_df_p = pd.DataFrame(selection_pvals).dropna()
    #display(selection_df_p)
    # Select significant features
    selection_selected = selection_df_p.sort_values("Pvalue", ascending=True).head(feat_num).loc[:, "Feature"].tolist()
    #print(selection_selected)
    X_train=pd.DataFrame(X_train, columns=X_train.columns)[selection_selected]
    X_test=pd.DataFrame(X_test, columns=X_test.columns)[selection_selected]

    # select highly variable features and drop correlated ones
    # unlikely to be used here as the features are already selected by stats but left just in case
    vt = VarianceThreshold(threshold=0.01)
    X_train_selected = vt.fit_transform(X_train)
    X_test_selected=vt.transform(X_test)

    X_train_selected=pd.DataFrame(X_train_selected, columns=vt.get_feature_names_out())
    X_test_selected=pd.DataFrame(X_test_selected, columns=vt.get_feature_names_out())

    print(f"After removing low variance features: {X_train_selected.shape[1]} features")

    # drop correlated features
    dcf = DropCorrelatedFeatures(threshold=0.7)
    X_train_selected=dcf.fit_transform(X_train_selected)
    X_test_selected = dcf.transform(X_test_selected)
    
    selected_features=X_train_selected.columns
    #print(selected_features)
    print(f"After removing correlated and low variance features: {X_train_selected.shape[1]} features")
    
    # Fit Cox model with standardization
    model = make_pipeline(StandardScaler(),RandomSurvivalForest(random_state=42))
    print(X_train_selected.shape, y_train.shape)

    # make hyperparameter grid and run gridsearchcv
    cv_param_grid = {
    "randomsurvivalforest__max_depth": np.array([1,2,5,10], dtype=int),
        "randomsurvivalforest__min_samples_split": np.array([5,10,20], dtype=int),
        "randomsurvivalforest__min_samples_leaf": np.array([2,5,10], dtype=int),
        "randomsurvivalforest__n_estimators": np.array([10, 20, 50, 100], dtype=int),
}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    event_indicator = y_train["Status"]
    gcv = GridSearchCV(
        estimator=model,
        param_grid=cv_param_grid,
        cv=cv.split(X_train_selected, event_indicator),
        n_jobs=-1
    ).fit(X_train_selected, y_train)
    
    final_model = gcv.best_estimator_
    print(final_model)
    final_model.fit(X_train_selected, y_train)
    
    # permutation based feature importance
    importance=permutation_importance(final_model, X_test_selected, y_test, random_state=42)

    feature_importance_df=pd.DataFrame(
    {
        k: importance[k]
        for k in (
            "importances_mean",
            "importances_std",
        )
    },
    index=X_test_selected.columns,
    ).sort_values(by="importances_mean", ascending=False)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.barh(
    feature_importance_df.index,
    feature_importance_df["importances_mean"],
    xerr=feature_importance_df["importances_std"],
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Mean Feature Importance")
    plt.title("Feature Importances with Standard Deviation")
    plt.tight_layout()
    plt.show()
    # ==============================================================================
    # 3. CONCORDANCE INDEX (C-INDEX) + IMPORTANCE STABILITY
    # ==============================================================================
    print("\n" + "="*80)
    print("3. CONCORDANCE INDEX EVALUATION + COEFFICIENT STABILITY")
    print("="*80)
    
    # Training C-index
    train_risk = final_model.predict(X_train_selected)
    train_c_index_censored = concordance_index_censored(y_train['Status'], y_train['Time'], train_risk)[0]
    
    # Test C-index
    test_risk = final_model.predict(X_test_selected)
    test_c_index_censored = concordance_index_censored(y_test['Status'], y_test['Time'], test_risk)[0]
    c_index_ipcw = concordance_index_ipcw(y_train, y_test, test_risk)[0]
    
    # Cross-validated C-index AND coefficient tracking
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    cv_scores = []
    perm_importance_dfs = []  # Create list of dfs with importances

    for i, (tr_idx, val_idx) in enumerate(cv.split(Xt, y["Status"]), 1):
        X_tr = Xt.iloc[tr_idx]
        X_val = Xt.iloc[val_idx]
        X_tr=normalise(X_tr).T
        X_val=normalise(X_val).T
        es = gp.gsva(data=X_tr.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
        X_tr=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
    
        es = gp.gsva(data=X_val.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
        X_val=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
        X_tr.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_") for col in X_tr.columns]
        X_val.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_") for col in X_val.columns]

        X_tr = X_tr[selected_features]
        X_val = X_val[selected_features]
    
        y_tr = y[tr_idx]
        y_val = y[val_idx]
    
        final_model.fit(X_tr, y_tr)
        val_risk = final_model.predict(X_val)
    
        c_index = concordance_index_censored(y_val["Status"], y_val["Time"], val_risk)[0]
        cv_scores.append(c_index)
        print(f"Fold {i}: C-index = {c_index:.4f}")


        importance=permutation_importance(final_model, X_test_selected, y_test, random_state=42)

        feature_importance_df=pd.DataFrame(
        {
            k: importance[k]
            for k in (
                "importances_mean",
                "importances_std",
            )
        },
        index=X_test_selected.columns,
        ).sort_values(by="importances_mean", ascending=False)
    
        # Extract coefficients from this fold
        perm_importance_dfs.append(feature_importance_df)
        
    cv_scores = np.array(cv_scores)
    
    print(f"CV C-index: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    print(f"\nTraining C-index: {train_c_index_censored:.4f}")
    print(f"Test C-index: {test_c_index_censored:.4f}")
    print(f"C-index IPCW: {c_index_ipcw}")
    print(f"Cross-validated C-index: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"CV scores: {cv_scores}")

    # ==============================================================================
    # 4. PERMUTATION IMPORTANCE STABILITY ACROSS FOLDS
    # ==============================================================================
    print("\n" + "="*80)
    print("4. PERMUTATION IMPORTANCE STABILITY ACROSS FOLDS")
    print("="*80)


# ------------------------------------------------------------------
# Stack fold-wise permutation importances
# ------------------------------------------------------------------
    selected_features_cv = perm_importance_dfs[0].index.tolist()

    # shape: (n_folds, n_features)
    cv_importances = np.vstack([
        df.loc[selected_features_cv, "importances_mean"].values
        for df in perm_importance_dfs
    ])

    # Mean & std across folds
    imp_mean = cv_importances.mean(axis=0)
    imp_std = cv_importances.std(axis=0)

    # ------------------------------------------------------------------
    # Stability dataframe
    # ------------------------------------------------------------------
    importance_stability_df = pd.DataFrame({
        "Feature": selected_features_cv,
        "Mean_Importance": imp_mean,
        "Std_Importance": imp_std,
        "CV_Importance": imp_std / (np.abs(imp_mean) + 1e-8)  # avoid div-by-zero
    }).sort_values("Mean_Importance", key=np.abs, ascending=False)

    print("\nPermutation Importance Stability Across Folds:")
    print(importance_stability_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Plot: Mean permutation importance ± SD
    # ------------------------------------------------------------------
    sort_idx = np.argsort(np.abs(imp_mean))[::-1]
    sorted_features = [selected_features_cv[i] for i in sort_idx]
    sorted_mean = imp_mean[sort_idx]
    sorted_std = imp_std[sort_idx]
    
    y_pos = np.arange(len(sorted_features))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(
        y_pos,
        sorted_mean,
        xerr=sorted_std,
        alpha=0.7,
        capsize=5,
        error_kw={'elinewidth': 2}
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f.replace('REACTOME_', '').replace('WP_', '').replace('GO_BP_', '')
         for f in sorted_features],
        fontsize=10
    )
    ax.set_xlabel("Permutation Importance (Mean ± SD)", fontsize=12)
    ax.set_title(
        "Permutation Feature Importance Across CV Folds",
        fontsize=14,
        fontweight="bold"
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Plot: Fold-wise permutation importances
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, feature_idx in enumerate(sort_idx):
        fold_imps = cv_importances[:, feature_idx]
    
        # Individual folds
        ax.scatter(
            [i] * len(fold_imps),
            fold_imps,
            alpha=0.6,
            s=100
        )
    
        # Mean
        ax.scatter(
            i,
            imp_mean[feature_idx],
            s=200,
            marker="D",
            color="black",
            edgecolors="white",
            linewidths=2,
            zorder=5
        )
    
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels(
        [f.replace('REACTOME_', '').replace('WP_', '').replace('GO_BP_', '')
         for f in sorted_features],
        rotation=45,
        ha="right",
        fontsize=10
    )
    ax.set_ylabel("Permutation Importance", fontsize=12)
    ax.set_title(
        "Permutation Importance Across Individual CV Folds\n"
        "(Diamond = Mean, Circles = Individual Folds)",
        fontsize=14,
        fontweight="bold"
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Plot: Coefficient of Variation (stability metric)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cv_sorted = importance_stability_df.sort_values("CV_Importance", ascending=False)
    
    ax.barh(
        range(len(cv_sorted)),
        cv_sorted["CV_Importance"],
        alpha=0.7,
        color="orange"
    )
    
    ax.set_yticks(range(len(cv_sorted)))
    ax.set_yticklabels(
        [f.replace('REACTOME_', '').replace('WP_', '').replace('GO_BP_', '')
         for f in cv_sorted["Feature"]],
        fontsize=10
    )
    ax.set_xlabel("Coefficient of Variation (Std / |Mean|)", fontsize=12)
    ax.set_title(
        "Permutation Importance Stability\n(Lower = More Stable)",
        fontsize=14,
        fontweight="bold"
    )
    ax.axvline(0.5, color="red", linestyle="--", linewidth=2, label="CV > 0.5 (Unstable)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()
    
    print("\nStability Assessment:")
    print(f"Features with CV > 0.5 (unstable): {(importance_stability_df['CV_Importance'] > 0.5).sum()}")
    print(f"Features with CV < 0.3 (stable): {(importance_stability_df['CV_Importance'] < 0.3).sum()}")

    # ==============================================================================
    # 5. RISK STRATIFICATION
    # ==============================================================================
    print("\n" + "="*80)
    print("5. RISK STRATIFICATION ANALYSIS")
    print("="*80)

    # calculate gsva but this time for the full data
    es = gp.gsva(data=Xt.T,
                         gene_sets=path_gmt,
                         kcdf="Gaussian", 
                         outdir=None)
    X_all=es.res2d.pivot(index='Term', columns='Name', values='ES').T.astype(float)
    X_all.columns=[col.replace(" ","_").replace("-","_").replace("(","_").replace(")","_").replace("&","_").replace(",","_").replace("/","_") for col in X_all.columns]
       
    # Calculate risk scores for all data
    all_risk = final_model.predict(X_all[selected_features])
    
    # Create risk groups (tertiles)
    risk_groups = pd.qcut(all_risk, q=3, labels=['Low', 'Medium', 'High'])
    
    # Count by group
    print("\nRisk Group Distribution:")
    print(risk_groups.value_counts().sort_index())
    
    # Event rates by group
    for group in ['Low', 'Medium', 'High']:
        mask = risk_groups == group
        event_rate = y[mask]['Status'].mean()
        median_time = np.median(y[mask]['Time'])
        print(f"{group} Risk: {mask.sum()} patients, {event_rate*100:.1f}% events, median time: {median_time:.1f}")
    
    # Kaplan-Meier curves by risk group
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    for group in ['Low', 'Medium', 'High']:
        mask = risk_groups == group
        kmf.fit(y[mask]['Time'], y[mask]['Status'], label=f'{group} Risk (n={mask.sum()})')
        kmf.plot_survival_function(ax=ax, color=colors[group], linewidth=2)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Kaplan-Meier Survival Curves by Risk Group', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Log-rank test
    logrank_result = multivariate_logrank_test(y['Time'], risk_groups, y['Status'])
    print(f"\nLog-rank test p-value: {logrank_result.p_value:.4e}")
    print(f"Test statistic: {logrank_result.test_statistic:.4f}")
    
    # ==============================================================================
    # 5. UNIVARIATE ANALYSIS OF EACH FEATURE
    # ==============================================================================
    print("\n" + "="*80)
    print("5. UNIVARIATE FEATURE ANALYSIS")
    print("="*80)

    # run univariate model without cv on the initial train-test split
    univariate_results = []
    for feature in X_train_selected.columns:
        # Fit univariate model
        uni_model = gcv.best_estimator_
        uni_model.fit(X_train_selected[[feature]], y_train)
        
        # Get metrics
        c_index = uni_model.score(X_test_selected[[feature]], y_test)
        
        univariate_results.append({
            'Feature': feature,
            'C_index': c_index
        })

    uni_df = pd.DataFrame(univariate_results).sort_values('C_index', ascending=False)
    print("\nUnivariate Cox Models:")
    print(uni_df.to_string(index=False))
    
    # Plot univariate C-indices
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(uni_df)), uni_df['C_index'], alpha=0.7, color='steelblue')
    ax.set_yticks(range(len(uni_df)))
    ax.set_yticklabels(uni_df['Feature'])
    ax.set_xlabel('C-index')
    ax.set_title('Univariate C-index for Each Feature')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1, label='Random')
    ax.axvline(test_c_index_censored, color='green', linestyle='--', linewidth=1, label='Multivariate Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

    # ==============================================================================
    # 6. RISK SCORE DISTRIBUTION
    # ==============================================================================
    print("\n" + "="*80)
    print("6. RISK SCORE DISTRIBUTION")
    print("="*80)
    
    # Separate by event status
    censored_risk = test_risk[y_test['Status'] == 0]
    event_risk = test_risk[y_test['Status'] == 1]
    
    print(f"\nRisk scores for censored: mean={censored_risk.mean():.3f}, std={censored_risk.std():.3f}")
    print(f"Risk scores for events: mean={event_risk.mean():.3f}, std={event_risk.std():.3f}")
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(event_risk, censored_risk)
    print(f"T-test p-value: {p_value:.4e}")
    
    # Plot distributions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(censored_risk, bins=30, alpha=0.6, label='Censored', color='blue', edgecolor='black')
    ax.hist(event_risk, bins=30, alpha=0.6, label='Event', color='red', edgecolor='black')
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Risk Score Distribution by Event Status', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ==============================================================================
    # 7. SUMMARY REPORT
    # ==============================================================================
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)

    summary = f"""
    Model Performance:
      - Training C-index: {train_c_index_censored:.4f}
      - Test C-index: {test_c_index_censored:.4f}
      - CV C-index: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
     
    
    Risk Stratification:
      - Log-rank test p-value: {logrank_result.p_value:.4e}
      - Significant separation between risk groups: {'Yes' if logrank_result.p_value < 0.05 else 'No'}

    """
    
    print(summary)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
