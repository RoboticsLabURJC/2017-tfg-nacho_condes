from labelmesequence import LabelMeSequence
import yaml
import utils
import pickle
from imageio import imread
import numpy as np

VIDEO_PATH = 'own_videos/test2'
TRUE10_PATH = 'test5/10_si'
FALSE10_PATH = 'test5/10_no'
TRUE20_PATH = 'test5/20_si'
FALSE20_PATH = 'test5/20_no'
OUT_PATH = 'test5/plot_matrix.csv'

if __name__ == '__main__':
    with open(TRUE10_PATH + '/benchmark.yml', 'r') as f:
        true10 = yaml.safe_load(f)['2.- Iterations']

    with open(FALSE10_PATH + '/benchmark.yml', 'r') as f:
        false10 = yaml.safe_load(f)['2.- Iterations']

    with open(TRUE20_PATH + '/benchmark.yml', 'r') as f:
        true20 = yaml.safe_load(f)['2.- Iterations']

    with open(FALSE20_PATH + '/benchmark.yml', 'r') as f:
        false20 = yaml.safe_load(f)['2.- Iterations']

    video = LabelMeSequence(VIDEO_PATH)

    plot_matrix = np.ones((len(video), 4)) * np.nan
    for idx, (image, labels) in enumerate(video):
        print(idx, labels)
        if '0' not in labels.keys():
            continue
        label = labels['0']
        for index in range(len(true10)):
            if true10[index]['1.- Frame'] == idx:
                rc = true10[index]['9.- RefCoords']
                x = rc['1.- X']
                y = rc['2.- Y']
                w = rc['3.- W']
                h = rc['4.- H']
                if x != '':
                    plot_matrix[idx, 0] = utils.jaccardIndex([float(x), float(y), float(w), float(h)], label)
                break
        for index in range(len(false10)):
            if false10[index]['1.- Frame'] == idx:
                rc = false10[index]['9.- RefCoords']
                x = rc['1.- X']
                y = rc['2.- Y']
                w = rc['3.- W']
                h = rc['4.- H']
                if x != '':
                    plot_matrix[idx, 1] = utils.jaccardIndex([float(x), float(y), float(w), float(h)], label)
                break
        for index in range(len(true20)):
            if true20[index]['1.- Frame'] == idx:
                rc = true20[index]['9.- RefCoords']
                x = rc['1.- X']
                y = rc['2.- Y']
                w = rc['3.- W']
                h = rc['4.- H']
                if x != '':
                    plot_matrix[idx, 2] = utils.jaccardIndex([float(x), float(y), float(w), float(h)], label)
                break
        for index in range(len(false20)):
            if false20[index]['1.- Frame'] == idx:
                rc = false20[index]['9.- RefCoords']
                x = rc['1.- X']
                y = rc['2.- Y']
                w = rc['3.- W']
                h = rc['4.- H']
                if x != '':
                    plot_matrix[idx, 3] = utils.jaccardIndex([float(x), float(y), float(w), float(h)], label)
                break

    np.savetxt(OUT_PATH, plot_matrix, delimiter=';')