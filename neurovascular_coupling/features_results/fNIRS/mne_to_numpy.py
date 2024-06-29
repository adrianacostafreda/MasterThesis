import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from DataPath import DataPath
from Hemo import HemoData
from sklearn.preprocessing import StandardScaler

def invertDic(dic: dict):
        '''
        Helping function to invert the keys and the values of a dictionary.
        '''
        inv_dic = {v: k for k, v in dic.items()}
        return inv_dic

def characterization_trigger_data(raw):
    
    # Dictionary to store trigger data
    trigger_data = {
        '0': {'begin': [], 'end': [], 'duration': []},
        '1': {'begin': [], 'end': [], 'duration': []},
        '2': {'begin': [], 'end': [], 'duration': []},
        '3': {'begin': [], 'end': [], 'duration': []},
        '4': {'begin': [], 'end': [], 'duration': []},
        '5': {'begin': [], 'end': [], 'duration': []},
        '6': {'begin': [], 'end': [], 'duration': []},
        '7': {'begin': [], 'end': [], 'duration': []}
    }

    annot = raw.annotations

    # Triggers 5, 6 & 7 --> 1-Back Test, 2-Back Test, 3-Back Test
    # Iterate over annotations 
    for idx, desc in enumerate(annot.description):
        if desc in trigger_data and desc not in ['0', '1', '2', '3']:  
            begin_trigger = annot.onset[idx] 
            duration_trigger = annot.duration[idx] + 50 # duration of 1, 2, 3 back is 60 s
            end_trigger = begin_trigger + duration_trigger

            # in case the file ends before
            if end_trigger >= raw.times[-1]:
                end_trigger = raw.times[-1]
            else:
                end_trigger = end_trigger
                
            #store trigger data in the dictionary
            trigger_data[desc]["begin"].append(begin_trigger)
            trigger_data[desc]["end"].append(end_trigger)
            trigger_data[desc]["duration"].append(duration_trigger)

    nback = list(range(0,4))
    indexes_1 = [7,8,9,10]
    indexes_2 = [5,6,7,8]

    
    if len(annot) == 11:
        for j in indexes_1:
            if j==7:
                begin_relax_0back = annot.onset[j]
                duration_relax_0back= 15
                end_relax_0back = begin_relax_0back + duration_relax_0back

                trigger_data["0"]["begin"].append(begin_relax_0back)
                trigger_data["0"]["end"].append(end_relax_0back)
                trigger_data["0"]["duration"].append(duration_relax_0back)

            elif j==8:
                begin_relax_1back = annot.onset[j] - 15 - 8
                duration_relax_1back= 15
                end_relax_1back = begin_relax_1back + duration_relax_1back

                trigger_data["1"]["begin"].append(begin_relax_1back)
                trigger_data["1"]["end"].append(end_relax_1back)
                trigger_data["1"]["duration"].append(duration_relax_1back)

                begin_relax_2back = annot.onset[j] + 60
                duration_relax_2back= 15
                end_relax_2back = begin_relax_2back + duration_relax_2back

                trigger_data["2"]["begin"].append(begin_relax_2back)
                trigger_data["2"]["end"].append(end_relax_2back)
                trigger_data["2"]["duration"].append(duration_relax_2back)

            elif j==9:
                # Store trigger data in the dictionary
                begin_relax_3back = annot.onset[j] + 60
                duration_relax_3back= 15
                end_relax_3back = begin_relax_3back + duration_relax_3back

                trigger_data["3"]["begin"].append(begin_relax_3back)
                trigger_data["3"]["end"].append(end_relax_3back)
                trigger_data["3"]["duration"].append(duration_relax_3back)
        

    elif len(annot) == 9:
        for j in indexes_2:
            if j==5:
                begin_relax_0back = annot.onset[j]
                duration_relax_0back= 15
                end_relax_0back = begin_relax_0back + duration_relax_0back

                trigger_data["0"]["begin"].append(begin_relax_0back)
                trigger_data["0"]["end"].append(end_relax_0back)
                trigger_data["0"]["duration"].append(duration_relax_0back)

            elif j==6:
                begin_relax_1back = annot.onset[j] - 15 - 8
                duration_relax_1back= 15
                end_relax_1back = begin_relax_1back + duration_relax_1back

                trigger_data["1"]["begin"].append(begin_relax_1back)
                trigger_data["1"]["end"].append(end_relax_1back)
                trigger_data["1"]["duration"].append(duration_relax_1back)

                begin_relax_2back = annot.onset[j] + 60
                duration_relax_2back= 15
                end_relax_2back = begin_relax_2back + duration_relax_2back

                trigger_data["2"]["begin"].append(begin_relax_2back)
                trigger_data["2"]["end"].append(end_relax_2back)
                trigger_data["2"]["duration"].append(duration_relax_2back)

            elif j==7:
                # Store trigger data in the dictionary
                begin_relax_3back = annot.onset[j] + 60
                duration_relax_3back= 15
                end_relax_3back = begin_relax_3back + duration_relax_3back

                trigger_data["3"]["begin"].append(begin_relax_3back)
                trigger_data["3"]["end"].append(end_relax_3back)
                trigger_data["3"]["duration"].append(duration_relax_3back)
            
    print(trigger_data)
    #----------------------------------------------------------------------------------

    # Determine the start of trigger 4
    # to set the beginning on Trigger 4, we compute the beginning of Trigger 5 and we subtract the relax
    # period (15s), and the test time (48s) & instructions (8s) of 1-Back Test
    # we add 10 seconds to start the segmentation 10 seconds after the trigger

    begin_trigger4 = trigger_data["5"]["begin"][0] - 15 - 48 - 8
    trigger_data["4"]["begin"].append(begin_trigger4)
    trigger_data["4"]["duration"].append(48) # Duration of 0-Back Test is 48 s

    end_time = trigger_data["4"]["begin"][0] + 48
    trigger_data["4"]["end"].append(end_time)

    # Set new annotations for the current file
    onsets = []
    durations = []
    descriptions = []

    # Accumulate trigger data into lists
    for description, data in trigger_data.items():
        for onset, duration in zip(data["begin"], data["duration"]):
            onsets.append(onset)
            durations.append(duration)
            descriptions.append(description)

    # Create new annotations
    new_annotations = mne.Annotations(
    onsets,
    durations,
    descriptions,
    ch_names=None  # Set to None since annotations don't have associated channel names
    )
    
    raw = raw.set_annotations(new_annotations)
    raw.plot(block=True)
    plt.show()
    
    #----------------------------------------------------------------------------------
    
    
    events, _ = mne.events_from_annotations(raw) # Create events from existing annotations
    
    print("These are the events", events)
    
    event_id = {"0": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8}

    # Using this event array, we take continuous data and create epochs ranging from -0.5 seconds before to 60 seconds after each event for 5,6,7 and 48 seconds for 4 
    # In other words, an epoch comprises data from -0.5 to 60 seconds around 5, 6, 7 events. For event 4, the epoch will comprise data from -0.5 to 48 seconds. 
    # We will consider Fz, Cz, Pz channels corresponding to the mid frontal line
    # Define the time windows
    tmin = -0.5 
    tmax = 15

    tmin_60s = -0.5
    tmax_60s = 60

    tmin_48s = -0.5
    tmax_48s = 48

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True
    )

    # Create epochs for events  with a duration of -0.5 to 15 seconds for relax
    epochs_relax = mne.Epochs(
        raw,
        events,
        event_id={"0": 1, "1": 2, "2": 3, "3":4},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True
    )
    
    # Create epochs for events 5, 6, 7 with a duration of -0.5 to 60 seconds
    epochs_60s = mne.Epochs(
        raw,
        events,
        event_id={"5": 5, "6": 6, "7": 7},
        tmin=tmin_60s,
        tmax=tmax_60s,
        baseline=None,
        preload=True
    )

    # Create epochs for event 4 with a duration of -0.5 to 48 seconds
    epochs_48s = mne.Epochs(
        raw,
        events,
        event_id={"4": 4},
        tmin=tmin_48s,
        tmax=tmax_48s,
        baseline=None,
        preload=True
    )

    # Concatenate the epochs
    #epochs = mne.concatenate_epochs([epochs_relax, epochs_48s, epochs_60s])
    
    return epochs

path_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
#path = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\patients\\baseline\\snirf_files\\"
path_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/patients/"

datapath_hc = DataPath(path_hc, recursive=False)
datapath_p = DataPath(path_p, recursive=False)

# Loop over all files preprocess them with HemoData and save them at an numpy array.
# It save the data for each patient, for each event, for each channel:  patient X events X channels X time_samples.
for id, file in enumerate(datapath_hc.getDataPaths()):
        raw_haemo = HemoData(file, preprocessing=True, isPloting=False).getMneIoRaw()
        
        epochs = characterization_trigger_data(raw_haemo)

        # --------------Epoched Data--------------------------------
        # Each epoched object contains 1s epoched data corresponding to each data segment

        raw_data_hbo = epochs.get_data(picks=["hbo"]) # epochs x channels x samples
        raw_data_hbr = epochs.get_data(picks=["hbr"]) # epochs x channels x samples

        # Initialize StandardScaler
        ss = StandardScaler()

        # Apply StandardScaler independently to each slice along the first dimension (axis=0)
        for i in range(raw_data_hbo.shape[0]):
            raw_data_hbo[i] = ss.fit_transform(raw_data_hbo[i])
            raw_data_hbr[i] = ss.fit_transform(raw_data_hbr[i])

        print("This is the shape of raw_data_hbo", raw_data_hbo.shape)

        raw_data_hbo = raw_data_hbo[:, :, :]
        raw_data_hbr = raw_data_hbr[:, :, :]

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_hc = np.expand_dims(raw_data_hbo[:, :, :],axis=0)
            data_hbr_hc = np.expand_dims(raw_data_hbr[:, :, :],axis=0)
        else:
            data_hbo_hc = np.concatenate((data_hbo_hc, np.expand_dims(raw_data_hbo[:, :, :],axis=0)),axis=0)
            data_hbr_hc = np.concatenate((data_hbr_hc, np.expand_dims(raw_data_hbr[:, :, :],axis=0)),axis=0)

for id, file in enumerate(datapath_p.getDataPaths()):
        raw_haemo = HemoData(file, preprocessing=True, useShortChannelRegression=True, isPloting=False).getMneIoRaw()
        #raw_haemo.plot(block=True)

        epochs = characterization_trigger_data(raw_haemo)

        # --------------Epoched Data--------------------------------
        # Each epoched object contains 1s epoched data corresponding to each data segment

        raw_data_hbo = epochs.get_data(picks=["hbo"]) # epochs x channels x samples
        raw_data_hbr = epochs.get_data(picks=["hbr"]) # epochs x channels x samples

        # Initialize StandardScaler
        ss = StandardScaler()

        # Apply StandardScaler independently to each slice along the first dimension (axis=0)
        for i in range(raw_data_hbo.shape[0]):
            raw_data_hbo[i] = ss.fit_transform(raw_data_hbo[i])
            raw_data_hbr[i] = ss.fit_transform(raw_data_hbr[i])

        raw_data_hbo = raw_data_hbo[:, :, :]
        raw_data_hbr = raw_data_hbr[:, :, :]

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_p = np.expand_dims(raw_data_hbo[:, :, :], axis=0)
            data_hbr_p = np.expand_dims(raw_data_hbr[:, :, :], axis=0)
        else:
            data_hbo_p = np.concatenate((data_hbo_p, np.expand_dims(raw_data_hbo[:, :, :], axis=0)),axis=0)
            data_hbr_p = np.concatenate((data_hbr_p, np.expand_dims(raw_data_hbr[:, :, :], axis=0)),axis=0)


def create_combined_plots(data_hbo_hc, data_hbo_p, raw):

    mean_subj_hc = np.mean(data_hbo_hc, axis=0)
    mean_subj_p = np.mean(data_hbo_p, axis=0)
    
    std_data_hbo_hc = np.std(data_hbo_hc, axis=0)
    std_data_hbo_p = np.std(data_hbo_p, axis=0)

    nback_0_hc = mean_subj_hc[4,:,:] - np.mean(mean_subj_hc[0, :,:], axis=1, keepdims=True)
    nback_1_hc = mean_subj_hc[5,:,:] - np.mean(mean_subj_hc[1,:,:], axis=1, keepdims=True)
    nback_2_hc = mean_subj_hc[6,:,:] - np.mean(mean_subj_hc[2,:,:], axis=1, keepdims=True)
    nback_3_hc = mean_subj_hc[7,:,:] - np.mean(mean_subj_hc[3,:,:], axis=1, keepdims=True)

    std_nback_0_hc = std_data_hbo_hc[4,:,:] - np.mean(std_data_hbo_hc[0, :,:], axis=1, keepdims=True)
    std_nback_1_hc = std_data_hbo_hc[5,:,:] - np.mean(std_data_hbo_hc[1,:,:], axis=1, keepdims=True)
    std_nback_2_hc = std_data_hbo_hc[6,:,:] - np.mean(std_data_hbo_hc[2,:,:], axis=1, keepdims=True)
    std_nback_3_hc = std_data_hbo_hc[7,:,:] - np.mean(std_data_hbo_hc[3,:,:], axis=1, keepdims=True)

    nback_0_p = mean_subj_p[4,:,:] - np.mean(mean_subj_p[0, :,:], axis=1, keepdims=True)
    nback_1_p = mean_subj_p[5,:,:] - np.mean(mean_subj_p[1,:,:], axis=1, keepdims=True)
    nback_2_p = mean_subj_p[6,:,:] - np.mean(mean_subj_p[2,:,:], axis=1, keepdims=True)
    nback_3_p = mean_subj_p[7,:,:] - np.mean(mean_subj_p[3,:,:], axis=1, keepdims=True)

    std_nback_0_p = std_data_hbo_p[4,:,:] - np.mean(std_data_hbo_p[0, :,:], axis=1, keepdims=True)
    std_nback_1_p = std_data_hbo_p[5,:,:] - np.mean(std_data_hbo_p[1,:,:], axis=1, keepdims=True)
    std_nback_2_p = std_data_hbo_p[6,:,:] - np.mean(std_data_hbo_p[2,:,:], axis=1, keepdims=True)
    std_nback_3_p = std_data_hbo_p[7,:,:] - np.mean(std_data_hbo_p[3,:,:], axis=1, keepdims=True)

    mean_data_hbo_hc = np.stack((nback_0_hc, nback_1_hc, nback_2_hc, nback_3_hc), axis=0)
    mean_data_hbo_p = np.stack((nback_0_p, nback_1_p, nback_2_p, nback_3_p), axis=0)

    std_data_hbo_hc = np.stack((std_nback_0_hc, std_nback_1_hc, std_nback_2_hc, std_nback_3_hc), axis=0)
    std_data_hbo_p = np.stack((std_nback_0_p, std_nback_1_p, std_nback_2_p, std_nback_3_p), axis=0)
    
    num_channels = mean_data_hbo_hc.shape[1]
    num_events = mean_data_hbo_hc.shape[0]
    
    time_hbo_hc = np.arange(data_hbo_hc.shape[-1])
    time_hbo_p = np.arange(data_hbo_p.shape[-1])

    # Determine global min and max for setting the same scale
    #global_min_hbo_hc = np.min(mean_data_hbo_hc - std_data_hbo_hc)
    #global_max_hbo_hc = np.max(mean_data_hbo_hc + std_data_hbo_hc)
    #global_min_hbo_p = np.min(mean_data_hbo_p - std_data_hbo_p)
    #global_max_hbo_p = np.max(mean_data_hbo_p + std_data_hbo_p)

    # Determine global min and max for setting the same scale
    global_min_hbo = min(np.min(mean_data_hbo_hc - std_data_hbo_hc), np.min(mean_data_hbo_p - std_data_hbo_p))
    global_max_hbo = max(np.max(mean_data_hbo_hc + std_data_hbo_hc), np.max(mean_data_hbo_p + std_data_hbo_p))


    fig, axes = plt.subplots(2, num_events, figsize=(20, 10))
    
    conditions = ['0-back', '1-back', '2-back', '3-back']
    
    # Define color map for regions
    region_colors = {
        "SMA": "blue",
        "Frontal": "green",
        "Motor Cortex": "red",
        "Parietal": "purple"
    }
    
    for i, condition in enumerate(conditions):
        
        # Plot HbO
        ax = axes[0, i]
        for channel in range(num_channels):
            if channel in (0, 1):
                region = "SMA"
            elif channel in (2, 3):
                region = "Frontal"
            elif channel in (4, 5):
                region = "Motor Cortex"
            elif channel in (6, 7):
                region = "Parietal"
                
            print(f"Channel {channel} corresponds to {region}")
            ax.plot(time_hbo_hc, mean_data_hbo_hc[i, channel, :], color=region_colors[region], alpha=0.5, label=f'{region} {channel}' if i == 0 else "")
            
        mean_vals_hbo_channels_hc = np.mean(mean_data_hbo_hc[i], axis=0)
        ax.plot(time_hbo_hc, mean_vals_hbo_channels_hc, 'k', label='Mean')
        ax.set_ylim(global_min_hbo, global_max_hbo)
        ax.set_xlim(0, data_hbo_hc.shape[-1])
        ax.set_title(f'{condition}')
            
        # Plot HbR
        ax = axes[1, i]
        for channel in range(num_channels):
            if channel in (0, 1):
                region = "SMA"
            elif channel in (2, 3):
                region = "Frontal"
            elif channel in (4, 5):
                region = "Motor Cortex"
            elif channel in (6, 7):
                region = "Parietal"
            else:
                region = "Unknown"

            print(f"Channel {channel} corresponds to {region}")
            ax.plot(time_hbo_p, mean_data_hbo_p[i, channel, :], color=region_colors[region],  alpha=0.5, label=f'{region} {channel}' if i == 0 else "")
            
        mean_vals_hbr_channels_p = np.mean(mean_data_hbo_p[i], axis=0)
        ax.plot(time_hbo_p, mean_vals_hbr_channels_p, 'k', label='Mean')
        ax.set_ylim(global_min_hbo, global_max_hbo)
        ax.set_xlim(0, data_hbo_p.shape[-1])
            
        ax.set_xlabel('Time in seconds')
    
    # Create custom legend
    legend_patches = [Patch(color=color, label=region) for region, color in region_colors.items()]
    fig.legend(handles=legend_patches, loc='upper right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Adding topomap for HbO and HbR
    fig, axes = plt.subplots(2, num_events, figsize=(20, 10))
    fig.suptitle('HbO Topomaps', fontsize=20)

    ch_idx_by_type = mne.channel_indices_by_type(raw.info)
    vmin = global_min_hbo
    vmax = global_max_hbo
    cmap = 'RdBu_r'

    ch_idx_by_type = mne.channel_indices_by_type(raw.info)
    
    for i, condition in enumerate(conditions):
        # HbO topomap Healthy Controls
        ax = axes[0, i]
        im, _ = mne.viz.plot_topomap(np.mean(mean_data_hbo_hc[i], axis=-1), mne.pick_info(raw.info, sel=ch_idx_by_type["hbo"]), axes=ax, show=False, vlim=(vmin,vmax), cmap=cmap)
        ax.set_title(f'HbO {condition} - Healthy Controls')
            
        # HbO topomap Patients
        ax = axes[1, i]
        im, _ = mne.viz.plot_topomap(np.mean(mean_data_hbo_p[i], axis=-1), mne.pick_info(raw.info, sel=ch_idx_by_type["hbo"]), axes=ax, show=False, vlim=(vmin,vmax), cmap=cmap)
        ax.set_title(f'HbO {condition} - Patients')

    # Add a single color bar for all topomaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    # Create custom legend for topomaps
    topomap_legend_patches = [Patch(color='white', label='HbO Mean Value')]

    fig.legend(handles=topomap_legend_patches, loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

# Function to create subplots for HbO with means across regions
def create_plots_mean_channels(data_hbo, title):
    regions = {
    'SMA': [0, 1],
    'Frontal': [2, 3],
    'Motor': [4, 5],
    'Parietal': [6, 7]}

    mean_subj = np.mean(data_hbo_hc, axis=0)
    std_data_hbo = np.std(data_hbo_hc, axis=0)


    nback_0 = mean_subj[4,:,:] - np.mean(mean_subj[0, :,:], axis=1, keepdims=True)
    nback_1 = mean_subj[5,:,:] - np.mean(mean_subj[1,:,:], axis=1, keepdims=True)
    nback_2 = mean_subj[6,:,:] - np.mean(mean_subj[2,:,:], axis=1, keepdims=True)
    nback_3 = mean_subj[7,:,:] - np.mean(mean_subj[3,:,:], axis=1, keepdims=True)

    std_nback_0 = std_data_hbo[4,:,:] - np.mean(std_data_hbo[0, :,:], axis=1, keepdims=True)
    std_nback_1 = std_data_hbo[5,:,:] - np.mean(std_data_hbo[1,:,:], axis=1, keepdims=True)
    std_nback_2 = std_data_hbo[6,:,:] - np.mean(std_data_hbo[2,:,:], axis=1, keepdims=True)
    std_nback_3 = std_data_hbo[7,:,:] - np.mean(std_data_hbo[3,:,:], axis=1, keepdims=True)

    mean_data_hbo = np.stack((nback_0, nback_1, nback_2, nback_3), axis=0)
    std_data_hbo = np.stack((std_nback_0, std_nback_1, std_nback_2, std_nback_3), axis=0)
   
    num_events = mean_data_hbo.shape[0]
    time_hbo = np.arange(data_hbo.shape[-1])

    # Determine global min and max for setting the same scale
    global_min_hbo = np.min(mean_data_hbo - std_data_hbo)
    global_max_hbo = np.max(mean_data_hbo + std_data_hbo)

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle(f'{title}', fontsize=20)
    
    for i, (region_name, channels) in enumerate(regions.items()):
        ax = axes[i // 2, i % 2]
        
        region_mean = np.mean(mean_data_hbo[:, channels, :], axis=1)
        
        for condition in range(num_events):
            ax.plot(time_hbo, region_mean[condition, :], label=f'{condition}-back')
        
        ax.set_ylim(global_min_hbo, global_max_hbo)
        ax.set_xlim(0, data_hbo.shape[-1])
        ax.set_title(f'{region_name}')
        if i % 2 == 0:
            ax.set_ylabel('HbO')
        if i // 2 == 1:
            ax.set_xlabel('Time in seconds')
        ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Create plots for healthy controls & patients
create_combined_plots(data_hbo_hc, data_hbo_p, raw_haemo)

# Create plots for healthy controls & patients
create_plots_mean_channels(data_hbo_hc, 'Healthy Controls - Mean HbO across Regions')
create_plots_mean_channels(data_hbo_p, 'Patients - Mean HbO across Regions')


