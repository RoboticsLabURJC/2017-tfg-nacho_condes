import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns; sns.set()
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['FreeSans']


yolo_file = 'test1/yolo.csv'
ssd_file = 'test1/ssd.csv'


yolo = np.loadtxt(yolo_file, delimiter=';')
ssd = np.loadtxt(ssd_file, delimiter=';')


fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
axs[0].set_title('IoU with ground truth')
axs[0].set_xlim([1, ssd.shape[0]])
axs[0].set_ylim([0, 1])
axs[0].set_xlabel('# frame')
axs[0].set_ylabel('IoU')
axs[0].plot(yolo[:,0], linewidth=1)
axs[0].plot(ssd[:, 0], linewidth=1)
axs[0].legend(['YOLO', 'SSD'])

axs[1].set_title('Inference time')
axs[1].set_xlim([1, ssd.shape[0]])
# axs[1].set_ylim([0, 1])
axs[1].set_xlabel('# frame')
axs[1].set_ylabel('ms')
axs[1].plot(yolo[:,1], linewidth=1)
axs[1].plot(ssd[:, 1], linewidth=1)
axs[1].legend(['YOLO', 'SSD'])

fig.savefig('test1/test1.pdf')
