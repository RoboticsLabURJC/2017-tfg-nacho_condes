import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns; sns.set()
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['FreeSans']


faced_file = 'test2/faced.csv'
haar_file = 'test2/haar.csv'


faced = np.loadtxt(faced_file, delimiter=';')
haar = np.loadtxt(haar_file, delimiter=';')

print('faced:', np.nanmean(faced))
print('haar:', np.nanmean(haar))

fig, axs = plt.subplots(figsize=(6, 4.5))
axs.set_title('IoU with ground truth')
axs.set_xlim([1, faced.shape[0]])
axs.set_ylim([0, 1])
axs.set_xlabel('# frame')
axs.set_ylabel('IoU')
axs.plot(faced, linewidth=1)
axs.plot(haar, linewidth=1)
axs.legend(['faced', 'Haar cascade'])


fig.savefig('test2/test2.pdf')
