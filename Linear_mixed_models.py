# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def lmm(df, group_col, visit_col, patient_col, variable_col, other_covariates=None):

    print("\n" + "="*70)
    print(f"PERFORMING ANALYSIS OF {variable_col.upper()}")
    print("="*70+"\n")

    # Ensure proper data types
    df[group_col] = df[group_col].astype('category')
    df[visit_col] = df[visit_col].astype(float)
    df[patient_col] = df[patient_col].astype('category')
    
    print("Data structure:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nPatients per group:\n{df.groupby(group_col, observed=False)[patient_col].nunique()}")
    
    # Summary statistics by group and visit
    summary = df.groupby([group_col, visit_col], observed=False)[variable_col].agg(['mean', 'std', 'count'])
    print("\n" + "="*70)
    print("Summary Statistics by Group and Visit:")
    print("="*70)
    print(summary)

    print("\n" + "="*70)
    print("TRAJECTORIES")
    print("="*70+"\n")
    # Visualize trajectories
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual trajectories
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group]
        for patient in group_data[patient_col].unique():
            patient_data = group_data[group_data[patient_col] == patient]
            axes[0].plot(patient_data[visit_col], patient_data[variable_col], 
                        alpha=0.3, linewidth=0.5)
    
    axes[0].set_xlabel('Visit')
    axes[0].set_ylabel(variable_col)
    axes[0].set_title('Individual Patient Trajectories')
    axes[0].legend(df[group_col].unique())
    
    # Mean trajectories with error bars
    for group in df[group_col].unique():
        group_summary = df[df[group_col] == group].groupby(visit_col)[variable_col].agg(['mean', 'sem'])
        axes[1].errorbar(group_summary.index, group_summary['mean'], 
                        yerr=group_summary['sem'], marker='o', label=group, capsize=5)
    
    axes[1].set_xlabel('Visit')
    axes[1].set_ylabel(variable_col)
    axes[1].set_title('Mean Trajectories by Group (±SEM)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

    # 3. LINEAR MIXED MODEL ANALYSIS

    print("\n" + "="*70)
    print("LINEAR MIXED MODEL ANALYSIS")
    print("="*70)
    
    # Model 1: Random intercept only (simplest model)
    # Formula: hematology_value ~ group + visit + group:visit
    # Random effect: patient_id (random intercept)
    covariate_str = ""
    if other_covariates:
        covariate_str = " + " + " + ".join(other_covariates)
        print(f"Including covariates: {', '.join(other_covariates)}\n")
    formula=f"{variable_col} ~ {group_col} + {visit_col} + {group_col}:{visit_col}{covariate_str}"
    
    model1 = mixedlm(f"{variable_col} ~ {group_col} + {visit_col} + {group_col}:{visit_col}{covariate_str}", 
                     df, 
                     groups=df[patient_col])
    result1 = model1.fit(method='lbfgs', maxiter=1000, reml=False)
    
    print("\nModel 1: Random Intercept Model")
    print("-" * 70)
    print("MODEL FORMULA")
    print(formula)
    print(result1.summary())
    
    # Model 2: Random intercept and random slope
    # This allows each patient to have their own trajectory slope
    
    
    model2 = mixedlm(f"{variable_col} ~ {group_col} + {visit_col} + {group_col}:{visit_col}{covariate_str}", 
                     df, 
                     groups=df[patient_col],
                     re_formula=f"~{visit_col}")
    result2 = model2.fit(method='lbfgs', maxiter=1000, reml=False)
    
    print("\n" + "="*70)
    print("Model 2: Random Intercept + Random Slope Model")
    print("-" * 70)
    print("MODEL FORMULA")
    print(formula)
    print(result2.summary())
    
    # Compare models using AIC/BIC
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"Model 1 (Random Intercept Only):")
    print(f"  AIC: {result1.aic:.2f}")
    print(f"  BIC: {result1.bic:.2f}")
    print(f"\nModel 2 (Random Intercept + Slope):")
    print(f"  AIC: {result2.aic:.2f}")
    print(f"  BIC: {result2.bic:.2f}")
    print(f"\nLower values indicate better fit. Preferred model: Model {'1' if result1.aic < result2.aic else '2'}")
    
    best_result = result2 if result2.aic < result1.aic else result1
    
    print("\n" + "="*70)
    print("KEY FINDINGS (from best model)")
    print("="*70)
    
    # Extract coefficients and p-values
    params = best_result.params
    pvalues = best_result.pvalues
    conf_int = best_result.conf_int()
    
    print("\nFixed Effects:")
    print("-" * 70)
    for param in params.index:
        print(f"{param:30s}: β = {params[param]:7.3f}, p = {pvalues[param]:.4f}, "
              f"95% CI [{conf_int.loc[param, 0]:.3f}, {conf_int.loc[param, 1]:.3f}]")
    
    # Interpretation helper
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
    1. Intercept: Mean hematology value for CTRL group at visit 0
    2. group[T.PD/PROD]: Difference from CTRL at baseline
    3. visit: Change per visit for CTRL group
    4. group[T.PD/PROD]:visit: Additional change per visit for PD/PROD vs CTRL
       (i.e., interaction effect - different slopes over time)
    """)
    
    # 6. MODEL DIAGNOSTICS
    print("\n" + "="*70)
    print("DIAGNOSTIC PLOTS")
    print("="*70+"\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Fitted
    fitted = best_result.fittedvalues
    residuals = best_result.resid
    
    axes[0, 0].scatter(fitted, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=30, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    
    # Scale-Location plot
    axes[1, 1].scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.5)
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('√|Residuals|')
    axes[1, 1].set_title('Scale-Location Plot')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    

   
