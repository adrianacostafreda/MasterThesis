# ========== Packages ==========
import mne, os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import pandas as pd
from statannotations.Annotator import Annotator
from basic.statistics import apply_stat_test

def plot_boxplot_band(df_psd, regions, band, condition_comp_list, condition_legend, fnt=['sans-serif',8,10],
    title=True, stat_test='Wilcoxon', ast_loc='inside', ylims=None, flier_label_xyloc=None, annot_offset=[0.1,0.1],
    yscale='linear', legend=False, figsize=(6,4), ylabel='PSD (µV\u00b2/Hz)', palette=None, verbose=True, export=False):
    """
    Plot boxplot for PSD values for a specific frequency band of interest at regions/channels.

    Parameters
    ----------
    df_psd: A Pandas dataframe of the power spectra values (for all the channels or regions)
    regions: A list of strings for all the regions/channels in interest (e.g. ['Fz','Fp1'])
    band: A string for the frequency band in interest (e.g. 'Alpha')
    condition_comp_list: A list of strings for experiment conditions codes to compare (e.g. [0-Back, 1-Back, 2-Back, 3-Back])
    condition_legend: A list of strings for the experiment conditions plotted
    fnt (optional): A list of font, and two font sizes (default: ['sans-serif',8,10])
    title (optional): A boolean for displaying the title of the plot
    stat_test (optional): A string for the statistical test for comparison (default: 'Wilcoxon')
    ast_loc (optional): A string for placement of asterix for statistical comparison (default: 'inside')
    ylims (optional): A list for y-scale limits (default: None)
    flier_label_xyloc (optional): Custom xy location for outlier label (in case they will be out of plot range)
    annot_offset (optional): Custom offset values for moving statistical test asterix annotations
    yscale (optional): Scale of y-scale (available: 'linear','log')
    legend (optional): Custom legend xy location (Matplotlib)
    figsize (optional): Figure size (Matplotlib)
    palette (optional): Figure color palette (Matplotlib/Seaborn)
    export (optional): A boolean for exporting, if True then the plot will be saved (default: False)
    """
    sns.set_style("whitegrid",{'font.family': [fnt[0]]})
    
    x = 'Region'
    hue = 'Condition'
    conditions = df_psd['Condition'].unique()
    
    df_psd_band = df_psd[df_psd['Frequency band'] == band]
    df_psd_band_final = pd.DataFrame()
    for region in regions:
        df_psd_temp = df_psd_band[[region,'Condition']].copy()
        df_psd_temp.rename(columns={region: band}, inplace=True)
        df_psd_temp['Region'] = region
        df_psd_band_final = pd.concat([df_psd_band_final,df_psd_temp]).reset_index(drop=True)
    
    fliercount = [0,0]
    if ylims != None:
        fliercount[0] = len(df_psd_band_final[band].values[df_psd_band_final[band] < ylims[0]])
        fliercount[1] = len(df_psd_band_final[band].values[df_psd_band_final[band] > ylims[1]])

    plt.figure(dpi=100,figsize = figsize)
    ax = sns.boxplot(x=x, y=band, hue=hue, data=df_psd_band_final, order=regions,
                     flierprops=dict(markerfacecolor = '0.5', markersize = 1),palette=palette)
    ax.set_yscale(yscale)

    pairs = []
    if stat_test=='t-test_paired' or stat_test=='Wilcoxon' or stat_test=='t-test_ind':
        for i in range(len(condition_comp_list)):
            _,_,_,significant = apply_stat_test(df_psd[df_psd['Frequency band']==band],condition_comp_list[i],stat_test=stat_test,verbose=verbose)
            for j in range(len(significant)):
                sign_temp = list(significant[j].values())[0]
                for s in range(len(regions)):
                    if sign_temp == regions[s]:
                        pairs.append(((*significant[j].values(),condition_comp_list[i][0]),(*significant[j].values(),condition_comp_list[i][1])))
    else:
        print(stat_test,'as a parameter for the statistical test is not supported.')

    if len(pairs) != 0:
        annotator = Annotator(ax,pairs=pairs,data=df_psd_band_final, x=x, y=band,
                        hue=hue,plot="boxplot",order=regions)\
                .configure(test=stat_test,text_format='star',loc=ast_loc,verbose=0)\
                .apply_test().annotate(line_offset_to_group=annot_offset[0], line_offset=annot_offset[1])
    
    if legend != False:
        if legend == True:
            kwargs = dict(loc = 'best')
        else:
            kwargs = legend
        plt.legend(title='Condition',title_fontsize=fnt[2],fontsize=fnt[1],**kwargs)
        for i in range(len(condition_legend)):
            ax.legend_.texts[i].set_text(condition_legend[i])
    else:
        ax.get_legend().remove()
    
    if title == True:
        if ast_loc == 'outside':
            ax.set_title('{} regional boxplot'.format(band),y=1.125,fontsize=fnt[2])
        else:
            ax.set_title('{} regional boxplot'.format(band),y=1.025,fontsize=fnt[2])
    
    if yscale == 'linear':
        ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
    plt.tick_params(axis='both', which='major', labelsize=fnt[2])
    plt.xlabel(x, fontsize=fnt[1])
    plt.ylabel(ylabel, fontsize=fnt[1])

    if ylims != None:
        ax.set(ylim=ylims)
        if flier_label_xyloc == None:
            flier_label_xyloc = [0,0]
            flier_label_xyloc[1] = 2
            if len(regions) == 1:
                flier_label_xyloc[0] = -0.05
            else:
                flier_label_xyloc[0] = (len(regions)-1)/2-0.15
        if fliercount[1] != 0:
            plt.text(flier_label_xyloc[0],ylims[1]-flier_label_xyloc[1],str(fliercount[1])+' outliers \u2191',size=fnt[1])
        if fliercount[0] != 0:
            plt.text(flier_label_xyloc[0],ylims[0]+flier_label_xyloc[1],str(fliercount[0])+' outliers \u2193',size=fnt[1])

    if export == True:
        try:
            os.makedirs(r"Results")
        except FileExistsError:
            pass
        plt.savefig('Results\psdboxplt_{}_{}_{}_{}.tiff'.format(band,regions,stat_test,conditions),dpi=300,bbox_inches='tight')
    
    plt.show()