import pandas as pd
import numpy  as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
    
    
def plot_t_evol(df, labels, variables, errors):
                     
    x_dates = pd.to_datetime(df["ts"], unit='s')
                     
    for var, err, label in zip(variables, errors, labels):
        y = df[var].values
        yerr = df[err].values if err in df.columns else [0] * len(df)
        
        rg = [y.min() - 0.1*abs(y.min()), y.max() + 0.1*abs(y.max())]

        fig, ax = plt.subplots(figsize=(10, 3))

        ax.errorbar(x_dates, y, yerr=yerr, fmt='o', color='black', capsize=0,
                    markersize=2.5, label='Monitoring')
        #ax.legend(loc='upper right')
        ax.set_ylabel(label)
        ax.set_ylim(rg)
        ax.grid(True)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m '))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        unique_runs = df.drop_duplicates(subset='run_number')
        run_times = pd.to_datetime(unique_runs["ts"], unit='s')

        for rt in run_times:
            ax.axvline(rt, color='red', linestyle='--', linewidth=1.2, alpha=0.6)

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        run_labels = unique_runs["run_number"].astype(str).tolist()
        ax_top.set_xticks(run_times)
        ax_top.set_xticklabels(run_labels, rotation=45, fontsize=8)

        plt.tight_layout()
        plt.show()