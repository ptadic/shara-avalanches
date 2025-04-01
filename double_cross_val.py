import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
import pandas as pd
from utils import (
    AvalancheFeatures, plot_relative_importances, load_labels, load_fahp_scores
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.inspection import permutation_importance
import scipy.stats as stats
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from joblib import dump, load
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

MODEL_NAMES = ('lr', 'dt', 'rf', 'l-svm', 'g-svm', 'mlp', 'lgbm', 'xgb')
METRIC_NAMES = ('accuracy', 'precision', 'recall', 'f1-score', 'auc', 'tprs')

# All ROC-curves are interpolated to a common x-axis (fpr)
COMMON_FPRS = np.linspace(0, 1)

# TODO docs

def train_lr(X, y):
    random_state = 0
    lr = LogisticRegressionCV(
        cv=5, random_state=random_state, solver='saga', penalty='l1', max_iter=3000
    ).fit(X, y)
    return lr

def train_dt(X, y):
    max_depths = [2, 4, 8, 16, None]
    min_samples_leaf = [1, 2, 5, 10]
    random_state = 0
    results = []
    for md in max_depths:
        for msl in min_samples_leaf:
            dtc = DecisionTreeClassifier(
                max_depth=md, min_samples_leaf=msl, random_state=random_state
            )
            result = cross_validate(dtc, X, y, cv=3)
            results.append({
                'max_depth': md,
                'min_samples_leaf': msl,
                'test_score_mean': result['test_score'].mean(),
                'test_score_std': result['test_score'].std()
            })
    results = pd.DataFrame(results)
    idx_best = results['test_score_mean'].idxmax()
    md = results.loc[idx_best, 'max_depth']
    md = None if np.isnan(md) else int(md)
    msl = results.loc[idx_best, 'min_samples_leaf']
    dt = DecisionTreeClassifier(
        max_depth=md, min_samples_leaf=msl, random_state=random_state
    ).fit(X, y)
    return dt

def train_lsvm(X, y):
    Cs = np.logspace(-1, 2, num=30)
    random_state = 0

    results = []
    for C in Cs:
        svc = SVC(C=C, random_state=random_state, kernel='linear', probability=True)
        result = cross_validate(svc, X, y, cv=3)
        results.append({
            'C': C,
            'test_score_mean': result['test_score'].mean(),
            'test_score_std': result['test_score'].std()
        })
    results = pd.DataFrame(results)
    idx_best = results['test_score_mean'].idxmax()
    C_best = results.loc[idx_best, 'C']
    svm = SVC(C=C_best, random_state=random_state, kernel='linear', probability=True).fit(
        X, y
    )
    return svm

def train_rf(X, y):
    n_estimators = [20, 50, 100]
    max_depths = [2, 4, 8, 16, None]
    random_state = 0
    results = []
    for ne in n_estimators:
        for md in max_depths:
            rf = RandomForestClassifier(
                n_estimators=ne, max_depth=md, oob_score=True, random_state=random_state
            ).fit(X, y)
            results.append({
                'n_estimators': ne, 'max_depth': md, 'oob_score': rf.oob_score_
            })
    results = pd.DataFrame(results)
    idx_best = results['oob_score'].idxmax()
    md = results.loc[idx_best, 'max_depth']
    md = None if np.isnan(md) else int(md)
    ne = results.loc[idx_best, 'n_estimators']
    rf = RandomForestClassifier(n_estimators=ne, max_depth=md, random_state=random_state)
    rf.fit(X, y)
    return rf

def train_gsvm(X, y):
    Cs = np.logspace(-1, 2, num=30)
    gammas = np.logspace(-3, 0, num=30)
    random_state = 0
    # Random sample of hyper-parameter combinations
    n_experiments = 100
    results = []
    random.seed(0)
    for i in range(n_experiments):
        C = random.choice(Cs)
        gamma = random.choice(gammas)
        svc = SVC(C=C, gamma=gamma, random_state=random_state, probability=True)
        result = cross_validate(svc, X, y, cv=3)
        results.append({
            'C': C, 'gamma': gamma, 
            'test_score_mean': result['test_score'].mean(),
            'test_score_std': result['test_score'].std()
        })
    results = pd.DataFrame(results)
    idx_best = results['test_score_mean'].idxmax()
    C_best = results.loc[idx_best, 'C']
    gamma_best = results.loc[idx_best, 'gamma']
    svm = SVC(
        C=C_best, gamma=gamma_best, random_state=random_state, probability=True
    ).fit(X, y)
    return svm

def train_mlp(X, y):
    max_layer_size = 100
    max_num_layers = 3
    random_state = 0

    # Random sample of hyper-parameter combinations
    n_experiments = 20
    results = []
    random.seed(0)
    for i in range(n_experiments):
        num_layers = random.randint(a=1, b=max_num_layers)
        layer_sizes = []
        for _ in range(num_layers):
            layer_sizes.append(random.randint(a=5, b=max_layer_size))
        mlp = MLPClassifier(
            hidden_layer_sizes=layer_sizes, 
            random_state=random_state, 
            max_iter=1000
        )
        result = cross_validate(mlp, X, y, cv=3)
        results.append({
            'layer_sizes': layer_sizes, 
            'test_score_mean': result['test_score'].mean(),
            'test_score_std': result['test_score'].std()
        })
    results = pd.DataFrame(results)
    idx_best = results['test_score_mean'].idxmax()
    layer_sizes_best = results.loc[idx_best, 'layer_sizes']
    mlp = MLPClassifier(
        hidden_layer_sizes=layer_sizes_best, 
        random_state=random_state, 
        max_iter=1000
    ).fit(X, y)
    return mlp

def train_lgbm(X, y):
    random_state = 0
    lgbm = LGBMClassifier(
        max_depth=8,
        min_data_in_leaf=4,
        random_state=random_state,
        verbose=-1
    ).fit(X, y)
    return lgbm

def train_xgb(X, y):
    random_state = 0
    param_dist = {
        'max_depth': stats.randint(3, 10),
        'learning_rate': stats.uniform(0.01, 0.5),
        'subsample': stats.uniform(loc=0.5, scale=0.5),
        'n_estimators':stats.randint(20, 200)
    }

    # Create the XGBoost model object
    xgb = XGBClassifier(random_state=random_state)

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        xgb, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy'
    )

    # Fit the RandomizedSearchCV object to the training data
    random_search.fit(X, y)
    xgb = XGBClassifier(
        n_estimators=150, 
        max_depth=7,
        subsample=0.9,
        learning_rate=0.5,     
        objective='binary:logistic',
        random_state=random_state
    )
    xgb.fit(X, y)
    return xgb


def train_model(model_name, X, y):

    if model_name == 'lr':
        # Logistic regression
        model = train_lr(X, y)

    elif model_name == 'dt':
        # Decision Tree
        model = train_dt(X, y)

    elif model_name == 'rf':
        # Random Forest
        model = train_rf(X, y)
    
    elif model_name == 'l-svm':
        # Support vector machine w/ linear kernel
        model = train_lsvm(X, y)

    elif model_name == 'g-svm':
        # Support vector machine w/ Gaussian (RBF) kernel
        model = train_lsvm(X, y)
        
    elif model_name == 'mlp':
        # Multi-Layer Perceptron 
        model = train_lsvm(X, y)
        
    elif model_name == 'lgbm':
        # LightGBM
        model = train_lgbm(X, y)

    elif model_name == 'xgb':
        # XGBoost
        model = train_xgb(X, y)
    
    else:
        model = None

    return model


def evaluate_model(model, X, y):
    # Make sure the order of columns in the design matrix
    # is the same as in the scaler:
    X = X[model.feature_names_in_]

    # Predict avalanche risk scores on test set
    scores = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    metrics = {}
    metrics['auc'] = roc_auc_score(y, scores)

    report = classification_report(y, predictions, output_dict=True, zero_division=np.nan)
    metrics['accuracy'] = report['accuracy']
    for k in ['precision', 'recall', 'f1-score']:
        metrics[k] = report['1'][k]

    # Get true and false positive ratios for ROC curves
    fpr, tpr, _ = roc_curve(y, scores)

    # Get rid of multiple points w/ same FPR value
    ind_last_zero = np.where(fpr == 0)[0][-1]
    fpr = fpr[ind_last_zero:]
    tpr = tpr[ind_last_zero:]

    # Interpolate to common x-axis
    tpr_interp = np.interp(COMMON_FPRS, fpr, tpr)

    metrics['tprs'] = tpr_interp 

    return metrics


def double_cross_eval(model_names, X, y, split_random_states):
    # Train and evaluate models over different train-test splits
    
    # Initialize structure to hold metrics for different models/metrics/splits
    metrics = {mdl: {} for mdl in model_names}
    agg_metrics = {mdl: {} for mdl in model_names}
    for mdl in model_names:
        for mtr in METRIC_NAMES:
            metrics[mdl][mtr] = []
            agg_metrics[mdl][mtr] = {}

    # Loop over different train/test splits
    for srs in split_random_states:
        print(f'Split random state = {srs}')
        # Train-test split
        idx_train, idx_test = train_test_split(
            range(len(y)), test_size=0.2, random_state=srs, stratify=y
        )        

        # Standardize
        ss = StandardScaler(copy=True).fit(X.iloc[idx_train])
        ss.set_output(transform='pandas')
        X_scaled = ss.transform(X)

        # Loop over models
        for mdl in model_names:

            # Train
            print(f'  Training {mdl}')    
            model = train_model(mdl, X_scaled.iloc[idx_train], y[idx_train])

            # Evaluate and store metrics
            print(f'  Evaluating {mdl}')
            metrics_ = evaluate_model(
                model, X_scaled.iloc[idx_test], y[idx_test]
            )
            for mtr in metrics_.keys():
                metrics[mdl][mtr].append(metrics_[mtr])
    
    # Aggregate metrics
    for mdl in model_names:
        for mtr in metrics[mdl].keys():
            agg_metrics[mdl][mtr]['mean'] = np.mean(metrics[mdl][mtr], axis=0)
            agg_metrics[mdl][mtr]['std'] = np.std(metrics[mdl][mtr], axis=0)

    return agg_metrics


def get_fahp_scores_predictions(fahp_scores_path, labels):
    # Load scores for the FAHP model
    fahp_scores = load_fahp_scores(
        tif_path=fahp_scores_path,
        coords=labels['geometry']
    )

    # Return value is GeoDataFrame w/ column 'fahp_score'
    fahp_scores = fahp_scores['fahp_score'].values

    # The range of these scores is 1-5; mapping to 0-1
    scores = (fahp_scores - 1.0) / 4.0

    # Range 2.93 - 3.56 define the 'average risk' bracket.
    # I'm using the mid point of this range 3.245 as the threshold for binary decision.
    predictions = (fahp_scores > 3.245).astype(int)

    return scores, predictions


def add_rocs_with_intervals(ax, agg_metrics, savefig_path=None, ):
    model_labels = {
        'g-svm': 'G-SVM', 'lgbm': 'LGBM', 'l-svm': 'L-SVM', 'lr': 'LR',
        'mlp': 'MLP', 'rf': 'RF', 'xgb': 'XGBoost', 'fahp': 'F-AHP', 'dt': 'Tree'
    }

    for i, mdl in enumerate(agg_metrics.keys()):
        # ROC
        label = f"{model_labels[mdl]}"
        label += f" (AUC = {100 * agg_metrics[mdl]['auc']['mean']:.1f}"
        label += f" $\pm$ {100 * agg_metrics[mdl]['auc']['std']:.1f})"
        tprs_mean = agg_metrics[mdl]['tprs']['mean']

        # Add the (0, 0) points, since it's missing from most ROCs
        fpr, tpr = np.r_[0, COMMON_FPRS], np.r_[0, tprs_mean]
        ax.plot(fpr, tpr, label=label, lw=2, alpha=0.8)
        
        # Trust region
        tprs_std = agg_metrics[mdl]['tprs']['std']
        hi = np.minimum(tprs_mean + tprs_std, 1)
        lo = np.maximum(tprs_mean - tprs_std, 0)
        ax.fill_between(COMMON_FPRS, lo, hi, color=f'C{i}', alpha=0.1)

    ax.set(
        xlabel='False positive rate', ylabel='True positive rate',
        title=r'Mean ROC curves with $\pm$ 1 std. dev. regions'
    )
    ax.grid()


def pretty_print_results(agg_metrics):
    data = {'model': agg_metrics.keys()}
    for mtr in [m for m in METRIC_NAMES if m != 'tprs']:
        data[mtr] = [f"{np.round(agg_metrics[mdl][mtr]['mean'] * 100, 1)} ({np.round(agg_metrics[mdl][mtr]['std'] * 100, 1)})" for mdl in agg_metrics.keys()]
    df = pd.DataFrame(data=data)
    print(df)


def make_roc_plots(agg_metrics, fahp_scores_path, labels, roc_path):
    # Create ROC plots
    fig, ax = plt.subplots(figsize=(6, 6))
    add_rocs_with_intervals(ax, agg_metrics)
    
    # Add FAHP ROC
    fahp_scores, _ = get_fahp_scores_predictions(fahp_scores_path, labels)
    fahp_auc = roc_auc_score(y, fahp_scores)
    fpr, tpr, _ = roc_curve(y, fahp_scores)
    ax.plot(fpr, tpr, label=f'F-AHP (AUC = {100 * fahp_auc:.1f})')

    # Random guess line
    ax.plot([0, 1], [0, 1], 'k--', label='Random guess')

    ax.legend(loc='lower right')

    # Save plots to disk
    fig.savefig(roc_path, bbox_inches='tight')

    # Zoomed upper-left corner
    ax.set(xlim=(-0.05, 0.3), ylim=(0.7, 1.05))
    ax.get_legend().remove()
    zoom_path = roc_path.with_stem(roc_path.stem + '_zoom')
    fig.savefig(zoom_path, bbox_inches='tight')

    plt.show()


def get_data(data_path, tif_file_paths):
    labels = load_labels(
        avalanches_path = data_path / 'Avalanches.shp',
        non_avalanches_path = data_path / 'Non avalanches.shp'
    )
    features = AvalancheFeatures(
        tif_paths = tif_file_paths,
        coords = labels['geometry']
    )

    # Get design matrix and label vector
    X = features.get_X()

    # Drop data points where some values are NaN
    nan_idx = X[X.isnull().any(axis=1)].index
    X = X.drop(index=nan_idx)
    labels = labels.drop(index = nan_idx)

    # Get label vector
    y = labels['avalanche'].values.astype(int)

    return labels, X, y


# TODO
# [X] Evaluation function
# [X] Train w/ 5 different train-test splits and compute metrics + ROC curves for each run
# [X] Include FAHP in the metrics
# [X] Break model training into individual calls so that you can do 1 model
# [X] Think about proper ROC: keep (0, 0) but eliminate all but last (0, non-zero)

if __name__ == '__main__':
    root = Path('.')
    data_path = root / 'data'
    tif_file_paths = list(data_path.glob('*.tif'))
    roc_path = root / 'results' / 'rocs_with_intervals.svg'
    fahp_scores_path = root / 'results' / 'FAHP_sintezna.tif'
    agg_metrics_path = root / 'results' / 'agg_metrics.pkl'

    # Load and data from disk and prepare for learning
    labels, X, y = get_data(data_path, tif_file_paths)

    # Seeds used to initialize random state generators for train-test splitting
    states = range(5)

    # Double cross validation returns means and std for each metric
    # agg_metrics = double_cross_eval(['lr'], X, y, states)
    if agg_metrics_path.is_file():
        agg_metrics = load(agg_metrics_path)
    else:
        agg_metrics = double_cross_eval(MODEL_NAMES, X, y, states)
        dump(agg_metrics, agg_metrics_path)

    # Print table w/ evaluation results
    pretty_print_results(agg_metrics)

    # Create ROC plots
    make_roc_plots(agg_metrics, fahp_scores_path, labels, roc_path)
    
    pass