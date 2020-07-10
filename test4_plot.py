import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import rcParams
import seaborn as sns; sns.set()
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['FreeSans']


OUT_FILE_PATH = 'test4/distances_list.pkl'

with open(OUT_FILE_PATH, 'rb') as f:
    distances_list = pickle.load(f)

plot_matrix = np.zeros((len(distances_list), 2))

for idx, dic in enumerate(distances_list):
    if '0' in dic.keys():
        plot_matrix[idx, 0] = dic['0']
    else:
        plot_matrix[idx, 0] = np.nan
    if '1' in dic.keys():
        plot_matrix[idx, 1] = dic['1']
    else:
        plot_matrix[idx, 1] = np.nan


fig, axs = plt.subplots(figsize=(6, 4.5))
axs.set_title('Distances to the reference face')
axs.set_xlim([1, len(distances_list)])
axs.plot(plot_matrix[:, 0], linewidth=1)
axs.plot(plot_matrix[:, 1], linewidth=1)
# axs.set_ylim([0.6, 1])
axs.set_xlabel('# frame')
axs.set_ylabel('Distance')
axs.legend(['Reference person', 'Non-reference person'])

fig.savefig('test4/test4.pdf')


print(np.nanmean(plot_matrix, axis=0), np.nanstd(plot_matrix, axis=0))