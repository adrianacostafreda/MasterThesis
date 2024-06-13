import numpy as np
import matplotlib.pyplot as plt

data_path_hbo_0back_hc = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\healthy_controls\\0_back\\hemo_0back.npy"
data_path_hbo_1back_hc = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\healthy_controls\\1_back\\hemo_1back.npy"
data_path_hbo_2back_hc = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\healthy_controls\\2_back\\hemo_2back.npy"
data_path_hbo_3back_hc = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\healthy_controls\\3_back\\hemo_3back.npy"

data_path_hbo_0back_p = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\0_back\\hemo_0back.npy"
data_path_hbo_1back_p = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\1_back\\hemo_1back.npy"
data_path_hbo_2back_p = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\2_back\\hemo_2back.npy"
data_path_hbo_3back_p = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\3_back\\hemo_3back.npy"

# healthy controls
load_0back_hc = np.load(data_path_hbo_0back_hc) # hc x epochs x channels x samples
load_1back_hc = np.load(data_path_hbo_1back_hc) # hc x epochs x channels x samples
load_2back_hc = np.load(data_path_hbo_2back_hc) # hc x epochs x channels x samples
load_3back_hc = np.load(data_path_hbo_3back_hc) # hc x epochs x channels x samples

# patients
load_0back_p = np.load(data_path_hbo_0back_p) # patients x epochs x channels x samples
load_1back_p = np.load(data_path_hbo_1back_p) # patients x epochs x channels x samples
load_2back_p = np.load(data_path_hbo_2back_p) # patients x epochs x channels x samples
load_3back_p = np.load(data_path_hbo_3back_p) # patients x epochs x channels x samples



#0back 

length_0back = np.arange(load_0back_hc.shape[-1])


mid_frontal_hc_0back = np.mean(load_0back_hc[:, :, 0:4, :], axis=-2)
mean_hc_0back = np.mean(mid_frontal_hc_0back, axis =0)

mean_epochs_hc_0back = np.mean(mean_hc_0back, axis =0)
std_epochs_hc_0back = np.std(mean_hc_0back, axis =0)

mid_frontal_p_0back = np.mean(load_0back_p[:, :, 0:4, :], axis=-2)
mean_p_0back = np.mean(mid_frontal_p_0back, axis =0)

mean_epochs_p_0back = np.mean(mean_p_0back, axis =0)
std_epochs_p_0back = np.std(mean_p_0back, axis =0)

#1back 

length_1back = np.arange(load_1back_hc.shape[-1])

mid_frontal_hc_1back = np.mean(load_1back_hc[:, :, 0:4, :], axis=-2)
mean_hc_1back = np.mean(mid_frontal_hc_1back, axis =0)

mean_epochs_hc_1back = np.mean(mean_hc_1back, axis =0)
std_epochs_hc_1back = np.std(mean_hc_1back, axis =0)

mid_frontal_p_1back = np.mean(load_1back_p[:, :, 0:4, :], axis=-2)
mean_p_1back = np.mean(mid_frontal_p_1back, axis =0)

mean_epochs_p_1back = np.mean(mean_p_1back, axis =0)
std_epochs_p_1back = np.std(mean_p_1back, axis =0)

#2back 

length_2back = np.arange(load_2back_hc.shape[-1])

mid_frontal_hc_2back = np.mean(load_2back_hc[:, :, 0:4, :], axis=-2)
mean_hc_2back = np.mean(mid_frontal_hc_2back, axis =0)

mean_epochs_hc_2back = np.mean(mean_hc_2back, axis =0)
std_epochs_hc_2back = np.std(mean_hc_2back, axis =0)

mid_frontal_p_2back = np.mean(load_2back_p[:, :, 0:4, :], axis=-2)
mean_p_2back = np.mean(mid_frontal_p_2back, axis =0)

mean_epochs_p_2back = np.mean(mean_p_2back, axis =0)
std_epochs_p_2back = np.std(mean_p_2back, axis =0)

#3back 

length_3back = np.arange(load_3back_hc.shape[-1])

mid_frontal_hc_3back = np.mean(load_3back_hc[:, :, 0:4, :], axis=-2)
mean_hc_3back = np.mean(mid_frontal_hc_3back, axis =0)

mean_epochs_hc_3back = np.mean(mean_hc_3back, axis =0)
std_epochs_hc_3back = np.std(mean_hc_3back, axis =0)

mid_frontal_p_3back = np.mean(load_3back_p[:, :, 0:4, :], axis=-2)
mean_p_3back = np.mean(mid_frontal_p_3back, axis =0)

mean_epochs_p_3back = np.mean(mean_p_3back, axis =0)
std_epochs_p_3back = np.std(mean_p_3back, axis =0)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 0back
axs[0, 0].plot(length_0back, mean_epochs_hc_0back, color="b", label=" HC Mean HbO Concentration")
#axs[0, 0].fill_between(length_0back, mean_epochs_hc_0back - std_epochs_hc_0back, mean_epochs_hc_0back + std_epochs_hc_0back, color='b', alpha=0.2, label="Standard Deviation")
axs[0, 0].plot(length_0back, mean_epochs_p_0back, color="g", label="P HbO Concentration")
#axs[0, 0].fill_between(length_0back, mean_epochs_p_0back - std_epochs_p_0back, mean_epochs_p_0back + std_epochs_p_0back, color='g', alpha=0.2, label="Standard Deviation")
axs[0, 0].set_xlabel("samples")
axs[0, 0].set_ylabel("hbo concentration")
axs[0, 0].set_title("0back")
axs[0, 0].legend()

# 1back
axs[0, 1].plot(length_1back, mean_epochs_hc_1back, color="b", label=" HC Mean HbO Concentration")
#axs[0, 1].fill_between(length_1back, mean_epochs_hc_1back - std_epochs_hc_1back, mean_epochs_hc_1back + std_epochs_hc_1back, color='b', alpha=0.2, label="Standard Deviation")
axs[0, 1].plot(length_1back, mean_epochs_p_1back, color="g", label="P HbO Concentration")
#axs[0, 1].fill_between(length_1back, mean_epochs_p_1back - std_epochs_p_1back, mean_epochs_p_1back + std_epochs_p_1back, color='g', alpha=0.2, label="Standard Deviation")
axs[0, 1].set_xlabel("samples")
axs[0, 1].set_ylabel("hbo concentration")
axs[0, 1].set_title("1back")
axs[0, 1].legend()

# 2back
axs[1, 0].plot(length_2back, mean_epochs_hc_2back, color="b", label=" HC Mean HbO Concentration")
#axs[1, 0].fill_between(length_2back, mean_epochs_hc_2back - std_epochs_hc_2back, mean_epochs_hc_2back + std_epochs_hc_2back, color='b', alpha=0.2, label="Standard Deviation")
axs[1, 0].plot(length_2back, mean_epochs_p_2back, color="g", label="P HbO Concentration")
#axs[1, 0].fill_between(length_2back, mean_epochs_p_2back - std_epochs_p_2back, mean_epochs_p_2back + std_epochs_p_2back, color='g', alpha=0.2, label="Standard Deviation")
axs[1, 0].set_xlabel("samples")
axs[1, 0].set_ylabel("hbo concentration")
axs[1, 0].set_title("2back")
axs[1, 0].legend()

# 3back
axs[1, 1].plot(length_3back, mean_epochs_hc_3back, color="b", label=" HC Mean HbO Concentration")
#axs[1, 1].fill_between(length_3back, mean_epochs_hc_3back - std_epochs_hc_3back, mean_epochs_hc_3back + std_epochs_hc_3back, color='b', alpha=0.2, label="Standard Deviation")
axs[1, 1].plot(length_3back, mean_epochs_p_3back, color="g", label="P HbO Concentration")
#axs[1, 1].fill_between(length_3back, mean_epochs_p_3back - std_epochs_p_3back, mean_epochs_p_3back + std_epochs_p_3back, color='g', alpha=0.2, label="Standard Deviation")
axs[1, 1].set_xlabel("samples")
axs[1, 1].set_ylabel("hbo concentration")
axs[1, 1].set_title("3back")
axs[1, 1].legend()

plt.tight_layout()
plt.show()