import data_gather
import numpy as np
import matplotlib.pyplot as plt

# gathering the images/labels
images, labels = data_gather.data_gather()

# conversion to np arrays
images = np.array(images)
#labels = np.array(labels)
'''
print len(labels)
# plotting
plt.hist(labels,30)
plt.title('Distribution of Traffic Sign Labels')

plt.show()

# viewing some signs
traffic_signs = [300, 2250, 3650, 4000]

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
                                                  images[traffic_signs[i]].min(),
                                                  images[traffic_signs[i]].max()))
plt.show()
'''
# now we will show a plot. For each label (class, total of 61), we will print how many signs for that class are there, and show the first one we encountered of that class on the dataset

unique_labels = set(labels) # ordered and unique labels
plt.figure(figsize=(15,15))
# counter
i = 1

for label in unique_labels:
    image = images[labels.index(label)]
    plt.subplot(8, 8, i)
    plt.axis('off')
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    i += 1
    plt.imshow(image)

plt.show()
