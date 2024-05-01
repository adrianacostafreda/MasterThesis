from DataPath import DataPath
import mne
from Hemo import HemoData
import matplotlib.pyplot as plt

mne.viz.set_browser_backend('qt')

path = "/Users/adriana/Documents/DTU/thesis/data_acquisition/fNIRS/healthy_controls/"
datapath = DataPath(path, recursive=False)
print(datapath.getDataPaths())
for file in datapath.getDataPaths():
    raw_haemo = HemoData(file, preprocessing=True)
    raw_haemo.plot(show=True)