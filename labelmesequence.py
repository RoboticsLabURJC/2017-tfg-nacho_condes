import json
import os
import cv2
import numpy as np
import utils

class LabelMeSequence():

    def __init__(self, root_dir):
        '''Fetches the (image, labels) pair from a ground truth video. The
        labels must have been created using the tool LabelMe, containing
        one rectangular label for each class, representing a person in the scene.

        Args:
            root_dir (string): directory containing one pair of files (x.jpg, x.json) for each image.
        '''
        # Get video labels
        self.root_dir = root_dir

    def __len__(self):
        files = os.listdir(self.root_dir)
        return len([file for file in files if file.endswith('.jpg')])

    def __getitem__(self, idx):
        idxnum = str(idx+1).zfill(4)
        # print(idxnum)
        # idxnum = idx

        image = cv2.imread(os.path.join(self.root_dir, f'{idxnum}.jpg'))
        if image is None:
            raise StopIteration
        try:
            with open(os.path.join(self.root_dir, f'{idxnum}.json'), 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {'shapes': []}
        instances = {}
        # Search for instances inside the label
        for inst in data['shapes']:
            label = inst['label']
            points = inst['points']
            if points[0][0] > points[1][0]:
                coords = list(map(int, points[1] + points[0]))
            else:
                coords = list(map(int, points[0] + points[1]))
            coords[0] = np.clip(coords[0], 0, image.shape[1])
            coords[1] = np.clip(coords[1], 0, image.shape[0])
            # coords = [int(point) for sub in inst['points'] for point in sub]
            coords = utils.corners2Corner(coords)
            instances[label] = coords

        return image, instances