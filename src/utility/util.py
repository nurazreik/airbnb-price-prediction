import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

def show_missing_values_info(df):
    ''' 
    Print the total number  and percentage of NA values w.r.t. features
    Print total number of samples that have a certain amount of NA values
    '''
    print("Total number of samples:", len(df))
    print("-" * 40)

    missing_cols = df.isna().sum()
    missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)

    missing_cols_pct = (missing_cols / len(df)) * 100

    missing_df = pd.DataFrame({
        'Number of NaN': missing_cols, 
        'Percentage (%)': missing_cols_pct.round(2)
    })

    print("MISSING VALUE w.r.t. FEATURE:")
    print(missing_df)
    print("-" * 40)

    missing_per_row = df.isna().sum(axis=1)

    print("MISSING VALUE w.r.t. SAMPLE:")
    row_missing_stats = missing_per_row.value_counts().sort_index()
    for missing_count, num_samples in row_missing_stats.items():
        if missing_count > 0:
            print(f"- There are {num_samples} rows (samples) with {missing_count} missing features")
    print()

def show_errors(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print("===== TEST SET =====")
    print(f"RMSE: €{rmse:.2f}")
    print(f"MAE:  €{mae:.2f}")
    print(f"Median AE:  €{medae:.2f}")
    print(f"R²:   {r2:.4f}")

    # R² on the log scale
    print(f"\nR² (log-scale): {r2_score(y_true_log, y_pred_log):.4f}")

def plot_distributions(df, columns):
    '''
    Plot distribution of given columns of a DataFrame.
    '''
    for col in columns:
        fig, ax = plt.subplots(figsize=(10, 5)) 
        
        sns.histplot(data=df, x=col, bins='auto', ax=ax)
        
        ax.set_title(f'Distribution: {col}')
        ax.set_ylabel('Frequency')
    
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        
        plt.xticks(fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
def plot_losses(losses, val_losses):
    plt.plot(losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend(); plt.xlabel('epoch'); plt.ylabel('loss')
    plt.show()

def plot_true_versus_predicted(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    lim = np.percentile(y_true, 99)
    plt.plot([0, lim], [0, lim], 'r--', label='perfect')
    plt.xlim(0, lim); plt.ylim(0, lim)
    plt.xlabel('Actual price (€)'); plt.ylabel('Predicted price (€)')
    plt.title('Predicted vs Actual — Test set'); plt.legend()
    plt.show()