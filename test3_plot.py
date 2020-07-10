import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns; sns.set()
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['FreeSans']


times_file = 'test3/times.csv'
JIs_file = 'test3/JIs.csv'


times = np.loadtxt(times_file, delimiter=';')
JIs = np.loadtxt(JIs_file, delimiter=';')

fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
axs[0].set_title(f'IoU of TensorRT with standard graph. Average value: {np.nanmean(JIs):.3f}')
axs[0].set_xlim([1, JIs.shape[0]])
axs[0].set_ylim([0.6, 1])
axs[0].set_xlabel('# frame')
axs[0].set_ylabel('IoU')
axs[0].plot(JIs)

axs[1].set_title('Inference time')
axs[1].set_xlim([1, times.shape[0]])
# axs[1].set_ylim([0, 1])
axs[1].set_xlabel('# frame')
axs[1].set_ylabel('ms')
axs[1].plot(times[:, 0], linewidth=1)
axs[1].plot(times[:, 1], linewidth=1)
axs[1].legend(['Standard graph', 'TensorRT graph'])

fig.savefig('test3/test3.pdf')


print('Original:', times[:, 0].mean(), times[:, 0].std())
print('TensorRT:', times[:, 1].mean(), times[:, 1].std())