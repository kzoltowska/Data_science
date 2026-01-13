# ============================
# FULL PIPELINE WITH PER-BRANCH SETTINGS
# ============================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, balanced_accuracy_score, roc_curve,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from feature_engine.selection import SmartCorrelatedSelection
warnings.filterwarnings('ignore')
from sklearn.utils import class_weight

# ----------------------------
# RANDOM SEED
# ----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Force CPU only (optional)
tf.config.set_visible_devices([], 'GPU')

# ----------------------------
# Smart HVG selection
# ----------------------------
def select_smart_hvg(X, y=None, top_n=300, corr_threshold=0.8):
    """
    Smart correlated feature selection + limit to top_n features
    """
    if y is not None:
        selector = SmartCorrelatedSelection(
            threshold=corr_threshold,
            selection_method='corr_with_target'
        )
        X_selected = selector.fit_transform(X, y)
    else:
        selector = SmartCorrelatedSelection(
            threshold=corr_threshold,
            selection_method='variance'
        )
        X_selected = selector.fit_transform(X)

    X_selected = pd.DataFrame(X_selected, columns=selector.get_feature_names_out())
    if top_n is not None and X_selected.shape[1] > top_n:
        top_vars = X_selected.var(axis=0).sort_values(ascending=False).head(top_n).index
        X_selected = X_selected.loc[:, top_vars]
    return X_selected

# ----------------------------
# CLS Token Layer
# ----------------------------
class CLSToken(layers.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.cls = self.add_weight(shape=(1,1,embed_dim), initializer="zeros", trainable=True)
    def call(self, x):
        batch = tf.shape(x)[0]
        cls = tf.tile(self.cls, [batch,1,1])
        return tf.concat([cls,x], axis=1)

# ----------------------------
# Transformer branch
# ----------------------------
def build_transformer_branch(num_tokens, embed_dim=32, num_heads=2, weight_decay=5e-4):
    inp = layers.Input(shape=(num_tokens,))
    x = layers.Reshape((num_tokens,1))(inp)
    x = layers.Dense(embed_dim)(x)
    x = CLSToken(embed_dim)(x)

    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)(x,x)
    x = layers.LayerNormalization()(x + attn)

    ff = layers.Dense(embed_dim*2, activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(x)
    ff = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(weight_decay))(ff)
    x = layers.LayerNormalization()(x + ff)

    cls_out = layers.Lambda(lambda t: t[:,0,:])(x)
    return Model(inp, cls_out)

# ----------------------------
# Projection head
# ----------------------------
def build_projection_head(embed_dim=32):
    inp = layers.Input(shape=(embed_dim,))
    x = layers.Dense(embed_dim, activation="relu")(inp)
    out = layers.Dense(embed_dim//2)(x)
    return Model(inp, out)

# ----------------------------
# Data augmentation
# ----------------------------
def augment_input(x, mask_ratio=0.1, noise_level=0.05):
    x_aug = x.copy()
    batch, tokens = x_aug.shape
    n_mask = max(1, int(tokens*mask_ratio))
    for i in range(batch):
        idx = np.random.choice(tokens, n_mask, replace=False)
        x_aug[i, idx] = 0.0
    x_aug += np.random.normal(0, noise_level, x_aug.shape)
    return x_aug.astype(np.float32)

# ----------------------------
# Contrastive loss
# ----------------------------
def contrastive_loss(z1, z2, temperature=0.3):
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    batch = tf.shape(z1)[0]
    if batch < 2: return tf.constant(0.0)
    z = tf.concat([z1,z2], axis=0)
    sim = tf.matmul(z,z,transpose_b=True)/temperature
    labels = tf.concat([tf.range(batch)+batch, tf.range(batch)], axis=0)
    mask = tf.eye(2*batch)
    sim = sim - mask*1e9
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, sim, from_logits=True)
    return tf.reduce_mean(loss)

# ----------------------------
# Pretrain branch
# ----------------------------
def pretrain_branch(X_ext, config):
    if isinstance(X_ext, pd.DataFrame):
        X_ext = X_ext.values.astype(np.float32)

    embed_dim = config.get('embed_dim', 32)
    num_heads = config.get('num_heads', 2)
    weight_decay = config.get('weight_decay', 5e-4)
    mask_ratio = config.get('mask_ratio', 0.1)
    noise_level = config.get('noise_level', 0.05)
    pretrain_epochs = config.get('pretrain_epochs', 50)
    pretrain_batch = config.get('pretrain_batch', 32)
    learning_rate = config.get('learning_rate', 5e-4)
    temperature = config.get('temperature', 0.3)

    branch = build_transformer_branch(X_ext.shape[1], embed_dim=embed_dim, num_heads=num_heads, weight_decay=weight_decay)
    projector = build_projection_head(embed_dim=embed_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(pretrain_epochs):
        idx = np.random.permutation(len(X_ext))
        losses = []

        for i in range(0, len(idx), pretrain_batch):
            batch_idx = idx[i:i+pretrain_batch]
            if len(batch_idx) < 2: continue
            x = X_ext[batch_idx]
            x1 = augment_input(x, mask_ratio=mask_ratio, noise_level=noise_level)
            x2 = augment_input(x, mask_ratio=mask_ratio, noise_level=noise_level)

            with tf.GradientTape() as tape:
                h1 = branch(x1, training=True)
                h2 = branch(x2, training=True)
                z1 = projector(h1)
                z2 = projector(h2)
                loss = contrastive_loss(z1, z2, temperature=temperature)

            grads = tape.gradient(loss, branch.trainable_variables + projector.trainable_variables)
            optimizer.apply_gradients(zip(grads, branch.trainable_variables + projector.trainable_variables))
            losses.append(loss.numpy())

        if (epoch + 1) % 5 == 0:
            print(f"Pretrain epoch {epoch+1}, loss={np.mean(losses):.4f}")

    return branch

# ----------------------------
# Triple-branch classifier
# ----------------------------
def build_triple_branch_model(num_genes, num_gsva1, num_gsva2,
                              gene_branch, gsva1_branch, gsva2_branch,
                              dropout_rate=0.6, embed_dim=32,
                              unfreeze_last_layers=False):
    gene_inp = layers.Input(shape=(num_genes,))
    gsva1_inp = layers.Input(shape=(num_gsva1,))
    gsva2_inp = layers.Input(shape=(num_gsva2,))

    gene_emb = gene_branch(gene_inp)
    gsva1_emb = gsva1_branch(gsva1_inp)
    gsva2_emb = gsva2_branch(gsva2_inp)

    # Freeze all layers first
    gene_branch.trainable = False
    gsva1_branch.trainable = False
    gsva2_branch.trainable = False

    if unfreeze_last_layers:
        # Unfreeze last Dense layer for fine-tuning
        if len(gene_branch.layers) >= 3: gene_branch.layers[-1].trainable = True
        if len(gsva1_branch.layers) >= 3: gsva1_branch.layers[-1].trainable = True
        if len(gsva2_branch.layers) >= 3: gsva2_branch.layers[-1].trainable = True

    x = layers.Concatenate()([gene_emb, gsva1_emb, gsva2_emb])
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(embed_dim*2, activation="relu",
                     kernel_regularizer=regularizers.l2(5e-4))(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model([gene_inp, gsva1_inp, gsva2_inp], out)

# ----------------------------
# Cross-validation
# ----------------------------
from imblearn.over_sampling import SMOTE
def cross_validate_model(X_counts, X_gsva1, X_gsva2, y,
                         gene_branch, gsva1_branch, gsva2_branch,
                         n_splits=3,
                         fine_tune_epochs=30,
                         fine_tune_batch=8,
                         use_youden=True,
                         unfreeze_last_layers=False,
                         return_models=True,
                         use_smote=True):
    """
    Cross-validation for triple-branch classifier with optional unfreezing and test set storage.

    Returns test sets per fold for feature importance analysis.
    """
  
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    metrics_list, roc_data = [], []
    y_trues, y_preds = [], []
    X_counts_test_list, X_gsva1_test_list, X_gsva2_test_list = [], [], []
    trained_models = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_counts, y)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")

        # Split data
        X_train_counts, X_test_counts = X_counts[train_idx], X_counts[test_idx]
        X_train_gsva1, X_test_gsva1 = X_gsva1.iloc[train_idx], X_gsva1.iloc[test_idx]
        X_train_gsva2, X_test_gsva2 = X_gsva2.iloc[train_idx], X_gsva2.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scaling
        scaler_counts = StandardScaler()
        X_train_counts = scaler_counts.fit_transform(X_train_counts)
        X_test_counts = scaler_counts.transform(X_test_counts)

        scaler_gsva1 = StandardScaler()
        X_train_gsva1 = scaler_gsva1.fit_transform(X_train_gsva1)
        X_test_gsva1 = scaler_gsva1.transform(X_test_gsva1)

        scaler_gsva2 = StandardScaler()
        X_train_gsva2 = scaler_gsva2.fit_transform(X_train_gsva2)
        X_test_gsva2 = scaler_gsva2.transform(X_test_gsva2)

        if use_smote:
            n_counts = X_train_counts.shape[1]
            n_gsva1 = X_train_gsva1.shape[1]
            n_gsva2 = X_train_gsva2.shape[1]

            # Concatenate scaled training features
            X_train_concat = np.hstack([
                X_train_counts,
                X_train_gsva1,
                X_train_gsva2
            ])

            smote = SMOTE(
                sampling_strategy="auto",
                k_neighbors=min(5, np.sum(y_train == 1) - 1),
                random_state=RANDOM_SEED
            )

            X_resampled, y_resampled = smote.fit_resample(
                X_train_concat, y_train
            )

            # Split back to branches
            X_train_counts = X_resampled[:, :n_counts]
            X_train_gsva1 = X_resampled[:, n_counts:n_counts + n_gsva1]
            X_train_gsva2 = X_resampled[:, n_counts + n_gsva1:]

            y_train = y_resampled

        # Build & compile model
        model = build_triple_branch_model(
            X_counts.shape[1], X_gsva1.shape[1], X_gsva2.shape[1],
            gene_branch, gsva1_branch, gsva2_branch, unfreeze_last_layers=unfreeze_last_layers
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy",
                      metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy")
    ])
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        # Class weights for imbalance
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = {i: w for i, w in enumerate(cw)}

        # Fit model
        model.fit([X_train_counts, X_train_gsva1, X_train_gsva2], y_train,
                  epochs=fine_tune_epochs,
                  batch_size=fine_tune_batch,
                  verbose=0,
                  callbacks=[early_stop],
                  class_weight=class_weights_dict)

        # Predict probabilities
        preds_prob = model.predict([X_test_counts, X_test_gsva1, X_test_gsva2]).ravel()

        # Compute Youden index threshold
        fpr, tpr, thresholds = roc_curve(y_test, preds_prob)
        if use_youden:
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]
        else:
            threshold = 0.5

        preds = (preds_prob >= threshold).astype(int)

        # Store metrics
        metrics_list.append({
            "AUC": roc_auc_score(y_test, preds_prob),
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "Balanced_Accuracy": balanced_accuracy_score(y_test, preds),
            "Youden_Threshold": threshold
        })

        roc_data.append((fpr, tpr, roc_auc_score(y_test, preds_prob)))
        y_trues.append(y_test)
        y_preds.append(preds)

        # Store test sets for feature importance
        X_counts_test_list.append(X_test_counts)
        X_gsva1_test_list.append(X_test_gsva1)
        X_gsva2_test_list.append(X_test_gsva2)

        if return_models:
            trained_models.append(model)

    if return_models:
        return (metrics_list, roc_data, y_trues, y_preds,
                trained_models,
                X_counts_test_list, X_gsva1_test_list, X_gsva2_test_list)
    
    return metrics_list, roc_data, y_trues, y_preds, X_counts_test_list, X_gsva1_test_list, X_gsva2_test_list


# ----------------------------
# Plot CV metrics
# ----------------------------
def plot_cv_results(all_metrics):
    metrics_df = pd.DataFrame(all_metrics)
    fig, axes = plt.subplots(2, 4, figsize=(18,10))
    metric_names = [m for m in metrics_df.columns if m != "Youden_Threshold"]
    for idx, metric in enumerate(metric_names):
        ax = axes[idx//4, idx%4]
        ax.bar(range(1,len(all_metrics)+1), metrics_df[metric], color='steelblue')
        ax.axhline(metrics_df[metric].mean(), color='red', linestyle='--',
                   label=f'Mean ± std: {metrics_df[metric].mean():.3f} ± {metrics_df[metric].std():.3f}')
        ax.set_xlabel('Fold'); ax.set_ylabel(metric); ax.set_title(metric)
        ax.legend(); ax.grid(True, alpha=0.3)

    # Youden thresholds
    ax = axes[1,3]
    ax.bar(range(1,len(all_metrics)+1), metrics_df["Youden_Threshold"], color='orange')
    ax.set_xlabel("Fold"); ax.set_ylabel("Youden Threshold"); ax.set_title("Optimal Youden Threshold per Fold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

# ----------------------------
# Plot ROC curves
# ----------------------------
def plot_roc_curves(roc_data):
    fig, ax = plt.subplots(figsize=(8,8))
    for i,(fpr,tpr,auc) in enumerate(roc_data):
        ax.plot(fpr, tpr, label=f'Fold {i+1} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1],'k--', label='Random')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curves')
    ax.legend(); ax.grid(True, alpha=0.3); plt.show()

# ----------------------------
# Plot confusion matrices
# ----------------------------
def plot_confusion_matrices(y_true_list, y_pred_list, fold_labels=None):
    n_folds = len(y_true_list)
    n_cols = 3
    n_rows = int(np.ceil(n_folds/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    axes = axes.flatten()
    for i in range(n_folds):
        cm = confusion_matrix(y_true_list[i], y_pred_list[i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f"Fold {i+1}" if fold_labels is None else fold_labels[i])
        axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("Actual")
    for j in range(n_folds, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout(); plt.show()

# ----------------------------
# SHAP feature importance across folds
# ----------------------------
import shap

def compute_shap_importance(trained_models,
                            X_counts_list, X_gsva1_list, X_gsva2_list,
                            feature_names_counts, feature_names_gsva1, feature_names_gsva2,
                            nsamples=50, top_n=20, use_gradient=True):
    """
    Compute SHAP feature importance across folds for triple-branch models.
    
    Parameters:
    -----------
    trained_models : list
        List of trained Keras models (one per fold)
    X_counts_list, X_gsva1_list, X_gsva2_list : list of arrays
        Input data for each branch across folds
    feature_names_* : array-like
        Feature names for each branch
    nsamples : int
        Number of samples for SHAP computation
    top_n : int
        Number of top features to return
    use_gradient : bool
        If True, use GradientExplainer (faster); if False, use KernelExplainer
    
    Returns:
    --------
    pd.DataFrame with columns: Feature, Mean_SHAP, SD_SHAP
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import shap
    
    all_shap_values = []
    all_feature_names = list(feature_names_counts) + list(feature_names_gsva1) + list(feature_names_gsva2)
    n_features = len(all_feature_names)
    
    for fold_idx, (model, Xc, X1, X2) in enumerate(zip(trained_models, X_counts_list, X_gsva1_list, X_gsva2_list)):
        print(f"Computing SHAP values for fold {fold_idx + 1}/{len(trained_models)}...")
        
        # Subsample for speed
        ns = min(nsamples, Xc.shape[0])
        indices = np.random.choice(Xc.shape[0], ns, replace=False)
        
        Xc_s = Xc[indices]
        X1_s = X1[indices]
        X2_s = X2[indices]
        
        if use_gradient:
            # GradientExplainer - much faster for neural networks
            explainer = shap.GradientExplainer(model, [Xc_s, X1_s, X2_s])
            shap_values = explainer.shap_values([Xc_s, X1_s, X2_s])
            
            print(f"  SHAP values type: {type(shap_values)}")
            print(f"  SHAP values structure: {type(shap_values[0]) if isinstance(shap_values, list) else 'not a list'}")
            
            # shap_values is a list of [shap_counts, shap_gsva1, shap_gsva2]
            # Each element corresponds to one input branch
            if isinstance(shap_values[0], list):
                # Multi-class: shap_values[branch][class]
                print(f"  Multi-class detected. Number of classes: {len(shap_values[0])}")
                
                # For each branch, average across classes, then concatenate branches
                shap_branches = []
                for branch_idx, branch_shap in enumerate(shap_values):
                    # branch_shap is a list of arrays (one per class)
                    # Stack them and take mean across classes
                    branch_abs = np.mean([np.abs(class_shap) for class_shap in branch_shap], axis=0)
                    shap_branches.append(branch_abs)
                    print(f"    Branch {branch_idx} shape after averaging: {branch_abs.shape}")
                
                shap_concat = np.hstack(shap_branches)
                
            else:
                # Binary/regression: shap_values is [array, array, array]
                print(f"  Binary/regression detected")
                shap_concat = np.hstack([
                    np.abs(shap_values[0]),  # counts branch
                    np.abs(shap_values[1]),  # gsva1 branch
                    np.abs(shap_values[2])   # gsva2 branch
                ])
                print(f"  Concatenated shape: {shap_concat.shape}")
                
        else:
            # KernelExplainer - slower but model-agnostic
            X_concat = np.hstack([Xc_s, X1_s, X2_s])
            
            def predict_fn(X_input):
                nc = Xc.shape[1]
                n1 = X1.shape[1]
                Xc_part = X_input[:, :nc]
                X1_part = X_input[:, nc:nc+n1]
                X2_part = X_input[:, nc+n1:]
                preds = model.predict([Xc_part, X1_part, X2_part], verbose=0)
                # Ensure 2D output for classifier
                if len(preds.shape) == 1:
                    preds = preds.reshape(-1, 1)
                return preds
            
            # Use smaller background for KernelExplainer
            background = shap.sample(X_concat, min(50, ns))
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values_raw = explainer.shap_values(X_concat, nsamples=100)
            
            print(f"  SHAP values type: {type(shap_values_raw)}")
            
            # Handle multi-class output
            if isinstance(shap_values_raw, list):
                # Average absolute SHAP values across classes
                print(f"  Multi-class detected. Number of classes: {len(shap_values_raw)}")
                shap_concat = np.mean([np.abs(sv) for sv in shap_values_raw], axis=0)
            else:
                shap_concat = np.abs(shap_values_raw)
            
            print(f"  Final concatenated shape: {shap_concat.shape}")
        
        # Validate shape
        print(f"  Expected features: {n_features}, Got: {shap_concat.shape[1]}")
        assert shap_concat.shape[1] == n_features, \
            f"SHAP shape mismatch: expected {n_features} features, got {shap_concat.shape[1]}"
        
        all_shap_values.append(shap_concat)
    
    # Aggregate across folds: compute mean and SD per feature
    # Stack: (n_folds * n_samples, n_features)
    all_shap_stacked = np.vstack(all_shap_values)
    print(f"\nStacked SHAP values shape: {all_shap_stacked.shape}")
    
    mean_shap = np.mean(all_shap_stacked, axis=0)
    sd_shap = np.std(all_shap_stacked, axis=0)
    
    print(f"Mean SHAP shape: {mean_shap.shape}")
    print(f"SD SHAP shape: {sd_shap.shape}")
    print(f"Number of features: {len(all_feature_names)}")
    
    # Ensure 1D arrays
    mean_shap = mean_shap.flatten()
    sd_shap = sd_shap.flatten()
    
    # Create results DataFrame
    df = pd.DataFrame({
        'Feature': all_feature_names,
        'Mean_SHAP': mean_shap,
        'SD_SHAP': sd_shap
    }).sort_values('Mean_SHAP', ascending=False).head(top_n)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    y_pos = np.arange(len(df))
    
    ax.barh(y_pos, df['Mean_SHAP'].values[::-1], 
            xerr=df['SD_SHAP'].values[::-1],
            color='steelblue', alpha=0.8, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Feature'].values[::-1])
    ax.set_xlabel('Mean |SHAP| Value', fontsize=12)
    ax.set_title('Top Feature Importances Across CV Folds (SHAP)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    
    return df
