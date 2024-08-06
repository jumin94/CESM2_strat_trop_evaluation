import xarray as xr
import numpy as np
import pandas as pd
import json 
import os
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
import glob
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib as mpl
import datetime

#Functions
#Funcion para calcular dia juliano
def julian(date):
    try:
        yday = datetime.utcfromtimestamp(date.tolist()/1e9).timetuple().tm_yday
    except AttributeError or TypeError:
        yday = date.timetuple().tm_yday
    return yday

def VB_date_ts(SPV_time_series):
    ts = np.array([])
    # Use a helper set to remove duplicates while preserving order
    seen = set()
    unique_list = []
    for item in SPV_time_series.time.dt.year.values:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)

        # Convert the unique list back to a NumPy array
    years = np.array(unique_list)
    for year in years:
        a = str(year)+'-10'
        b = str(year+1)+'-02'
        ua = SPV_time_series.sel(time=slice(a,b))
        T = ua.time
        for i in range(len(ua)):
            if not (ua[i] > 15): #Encuentra la fecha del breakdown
                vb1 = julian(T.values[i]) 
                #print(vb1,year)
                if vb1 < 200:
                    vb1 = vb1 + 365 
                else:
                    vb1 = vb1
                ts = np.append(ts,vb1) #Agrega al array
                break

    return ts

def normalize(data):
    return ((data - np.mean(data))/np.std(data))

def marshall_sam(psl_field):
    SAM_index = normalize(psl_field.sel(lat=-40,method='nearest').mean(dim='lon')) - normalize(psl_field.sel(lat=-65,method='nearest').mean(dim='lon'))
    return SAM_index

def prob_SAM(data,mean,std):
    p_pos = len(data[data > (mean + std)]) / len(data)
    p_neg = len(data[data < (mean - std)]) / len(data)
    return p_pos, p_neg
    

def probability_ONDJF(VB_dates):
    print(VB_dates)
    p_O = len(VB_dates[(274 <= VB_dates) & (VB_dates <= 304)]) / len(VB_dates)
    p_N = len(VB_dates[(305 <= VB_dates) & (VB_dates <= 334)]) / len(VB_dates)
    p_D = len(VB_dates[(335 <= VB_dates) & (VB_dates <= 365)]) / len(VB_dates)
    p_J = len(VB_dates[(366 <= VB_dates) & (VB_dates <= 397)]) / len(VB_dates)
    p_F = len(VB_dates[(398 <= VB_dates) & (VB_dates <= 426)]) / len(VB_dates)
    return [p_O,p_N,p_D,p_J,p_F]


def VB_date_distribution_plot(VB_dates,title):
    fig = plt.figure(figsize=(5,5),dpi=300)
    plt.hist(VB_dates,label='p(VB|month) = '+str(np.round(probability_ONDJF(VB_dates),2)))
    plt.xlabel('Julian day')
    plt.ylabel('Vortex Breakdown counts')
    plt.xlim(274,426)

    # Compute the mean of VB_dates
    mean_value = np.mean(VB_dates)

    # Shading and labeling for each month
    months = {
        'October': (274, 304),
        'November': (305, 334),
        'December': (335, 365),
        'January': (366, 397),
        'February': (398, 426)
    }

    colors = ['white', 'lightgrey']

    for i, (month, (start, end)) in enumerate(months.items()):
        plt.axvspan(start, end, facecolor=colors[i % 2], alpha=0.5)
        plt.text((start + end) / 2, plt.ylim()[1] * 0.95, month, 
                horizontalalignment='center', verticalalignment='top')

    # Add a dashed vertical line at the mean value of VB_dates
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_value:.2f}')
    
    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=5) 

    return fig

def SAM_distribution(SAM_index,VB_dates,title):
    fig = plt.figure(figsize=(5,5),dpi=300)
    std = np.std(SAM_index).values
    mean = np.mean(SAM_index).values
    plt.hist(SAM_index,alpha=0.5)
    SAM_earlyVB = SAM_index.values[VB_dates[:-1] < (np.mean(VB_dates) - np.std(VB_dates))]
    p_SAM_earlyVB = prob_SAM(SAM_earlyVB,mean,std)
    plt.hist(SAM_earlyVB,alpha=0.8,label='p(SAM+|earlyVB)= '+str(round(p_SAM_earlyVB[0],2))+'\n p(SAM-|earlyVB)= '+str(round(p_SAM_earlyVB[1],2)))
    SAM_lateVB = SAM_index.values[VB_dates[:-1] > (np.mean(VB_dates) + np.std(VB_dates))]
    p_SAM_lateVB = prob_SAM(SAM_lateVB,mean,std)
    plt.hist(SAM_lateVB,alpha=0.8,label='p(SAM+|lateVB)= '+str(round(p_SAM_lateVB[0],2))+'\n p(SAM-|lateVB)= '+str(round(p_SAM_lateVB[1],2)))
    plt.legend(fontsize=8)
    plt.xlabel('Marshall SAM index')
    plt.ylabel('DJF season count')
    plt.title(title)
    return fig


def marginal_distribution(SAM_index,VB_dates,title):
    SAM_earlyVB = SAM_index.values[VB_dates[:-1] < (np.mean(VB_dates) - np.std(VB_dates))]
    SAM_lateVB = SAM_index.values[VB_dates[:-1] > (np.mean(VB_dates) + np.std(VB_dates))]
    fig = plt.figure(figsize=(5,5),dpi=300)
    data = [SAM_earlyVB,SAM_lateVB]
    # Create a boxplot
    plt.boxplot(data, labels=['early VB', 'late VB'])

    # Add title and labels
    plt.title('Stratospheric-tropospheric coupling in \n '+title)
    plt.ylabel('Marshall SAM index')
    return fig


def save_to_csv(path,lista,name):
    df = {str(i): sim for i,sim in enumerate(lista)}
    pd.DataFrame(df).to_csv(path+'/'+name+'.csv')
    return 'saved index'
    
def main(config):
    """Run the diagnostic."""
    cfg=get_cfg(os.path.join(config["run_dir"],"settings.yml"))
    print(cfg)
    meta = group_metadata(config["input_data"].values(), "alias")
    #print(f"\n\n\n{meta}")
    for alias, alias_list in meta.items():
        #print(f"Computing index regression for {alias}\n")
        SPV = [xr.open_dataset(m["filename"])[m["short_name"]] for m in alias_list if m["variable_group"] == "ua50_spv"]
        PSL = [xr.open_dataset(m["filename"])[m["short_name"]] for m in alias_list if m["variable_group"] == "psl"]
        #Compute vortex breakdown date
        VB_dates = [VB_date_ts(spv_ts) for spv_ts in SPV]
        SAM_djf = [marshall_sam(psl_djf) for psl_djf in PSL]
        #Evaluate marginal probabilities for each month
        fig1 = VB_date_distribution_plot(VB_dates[0],alias)
        #Evaluate SAM index distribution and conditional probabilities
        fig2 = SAM_distribution(SAM_djf[0],VB_dates[0],alias)
        #Evaluate SAM index distribution and conditional probabilities
        fig3 = marginal_distribution(SAM_djf[0],VB_dates[0],alias)
        print(f"Save figures {alias}\n")
        os.chdir(config["work_dir"])
        os.getcwd()
        os.makedirs("indices",exist_ok=True)
        os.chdir(config["work_dir"]+'/'+"indices")
        os.makedirs(alias,exist_ok=True)
        save_to_csv(config["work_dir"]+'/indices/'+alias,VB_dates,'vortex_breakdown_dates')
        save_to_csv(config["work_dir"]+'/indices/'+alias,SAM_djf,'SAM_index')
        #Plot coefficients
        os.chdir(config["plot_dir"])
        os.getcwd()
        os.makedirs("strat_trop_coupling",exist_ok=True)
        fig1.savefig(config["plot_dir"]+'/strat_trop_coupling/VB_distribution'+alias+'.png')
        fig2.savefig(config["plot_dir"]+'/strat_trop_coupling/SAM_distribution'+alias+'.png')
        fig3.savefig(config["plot_dir"]+'/strat_trop_coupling/Boxplot'+alias+'.png')
          
if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
                                    
