import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams
import seaborn as sns; sns.set()
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['FreeSans']


OUT_PATH = 'test5/plot_matrix.csv'

plot_matrix = np.loadtxt(OUT_PATH, delimiter=';')

OCCLUSION_IDX = [720, 873]
occ_alpha = 0.1
linewidth = 0.4
n_points = plot_matrix.shape[0]
fig, axs = plt.subplots(2, 2, figsize=(16, 11.5))
fig.suptitle('IoU with ground truth')

axs[0][0].set_title('$k=10$')
axs[0][0].set_xlim([1, n_points])
axs[0][0].set_ylim([0, 1])
axs[0][0].plot(range(OCCLUSION_IDX[0]), plot_matrix[:OCCLUSION_IDX[0], 0], linewidth=linewidth, alpha=0.7, color='C0')
axs[0][0].plot(range(OCCLUSION_IDX[0]), plot_matrix[:OCCLUSION_IDX[0], 1], linewidth=linewidth, alpha=0.7, color='C1')
axs[0][0].plot(range(OCCLUSION_IDX[0], OCCLUSION_IDX[1]), plot_matrix[OCCLUSION_IDX[0]:OCCLUSION_IDX[1], 0], linewidth=1, color='C0')
axs[0][0].plot(range(OCCLUSION_IDX[0], OCCLUSION_IDX[1]), plot_matrix[OCCLUSION_IDX[0]:OCCLUSION_IDX[1], 1], linewidth=1, color='C1')
axs[0][0].plot(range(OCCLUSION_IDX[1], n_points), plot_matrix[OCCLUSION_IDX[1]:, 0], linewidth=linewidth, alpha=0.7, color='C0')
axs[0][0].plot(range(OCCLUSION_IDX[1], n_points), plot_matrix[OCCLUSION_IDX[1]:, 1], linewidth=linewidth, alpha=0.7, color='C1')
axs[0][0].legend(['with tracker', 'without tracker'])
axs[0][0].set_xlabel('# frame')
axs[0][0].set_ylabel('IoU')

axs[0][1].set_title('$k=20$')
axs[0][1].set_xlim([1, n_points])
axs[0][1].set_ylim([0, 1])
axs[0][1].plot(range(OCCLUSION_IDX[0]), plot_matrix[:OCCLUSION_IDX[0], 2], linewidth=linewidth, alpha=0.7, color='C0')
axs[0][1].plot(range(OCCLUSION_IDX[0]), plot_matrix[:OCCLUSION_IDX[0], 3], linewidth=linewidth, alpha=0.7, color='C1')
axs[0][1].plot(range(OCCLUSION_IDX[0], OCCLUSION_IDX[1]), plot_matrix[OCCLUSION_IDX[0]:OCCLUSION_IDX[1], 2], linewidth=1, color='C0')
axs[0][1].plot(range(OCCLUSION_IDX[0], OCCLUSION_IDX[1]), plot_matrix[OCCLUSION_IDX[0]:OCCLUSION_IDX[1], 3], linewidth=1, color='C1')
axs[0][1].plot(range(OCCLUSION_IDX[1], n_points), plot_matrix[OCCLUSION_IDX[1]:, 2], linewidth=linewidth, alpha=0.7, color='C0')
axs[0][1].plot(range(OCCLUSION_IDX[1], n_points), plot_matrix[OCCLUSION_IDX[1]:, 3], linewidth=linewidth, alpha=0.7, color='C1')
axs[0][1].legend(['with tracker', 'without tracker'])
axs[0][1].set_xlabel('# frame')
axs[0][1].set_ylabel('IoU')

axs[1][0].set_title('$k=10$ (occlusion lapse)')
axs[1][0].set_xlim(OCCLUSION_IDX)
axs[1][0].set_ylim([0, 1])
axs[1][0].plot(range(OCCLUSION_IDX[0], OCCLUSION_IDX[1]), plot_matrix[OCCLUSION_IDX[0]:OCCLUSION_IDX[1], 0], linewidth=1, color='C0')
axs[1][0].plot(range(OCCLUSION_IDX[0], OCCLUSION_IDX[1]), plot_matrix[OCCLUSION_IDX[0]:OCCLUSION_IDX[1], 1], linewidth=1, color='C1')
axs[1][0].legend(['with tracker', 'without tracker'])
axs[1][0].set_xlabel('# frame')
axs[1][0].set_ylabel('IoU')

axs[1][1].set_title('$k=20$ (occlusion lapse)')
axs[1][1].set_xlim(OCCLUSION_IDX)
axs[1][1].set_ylim([0, 1])
axs[1][1].plot(range(OCCLUSION_IDX[0], OCCLUSION_IDX[1]), plot_matrix[OCCLUSION_IDX[0]:OCCLUSION_IDX[1], 2], linewidth=1, color='C0')
axs[1][1].plot(range(OCCLUSION_IDX[0], OCCLUSION_IDX[1]), plot_matrix[OCCLUSION_IDX[0]:OCCLUSION_IDX[1], 3], linewidth=1, color='C1')
axs[1][1].legend(['with tracker', 'without tracker'])
axs[1][1].set_xlabel('# frame')
axs[1][1].set_ylabel('IoU')


fig.savefig('test5/test5.pdf')