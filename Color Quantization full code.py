import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

image_as_array = mpimg.imread('../DATA/palm_trees.jpg')
image_as_array # RGB CODES FOR EACH PIXEL

plt.figure(figsize=(6,6),dpi=200)
plt.imshow(image_as_array)

image_as_array.shape
# (h,w,3 color channels)

(h,w,c) = image_as_array.shape
image_as_array2d = image_as_array.reshape(h*w,c)
from sklearn.cluster import KMeans
model = KMeans(n_clusters=6)
model

labels = model.fit_predict(image_as_array2d)
labels

# THESE ARE THE 6 RGB COLOR CODES
model.cluster_centers_

rgb_codes = model.cluster_centers_.round(0).astype(int)
rgb_codes

quantized_image = np.reshape(rgb_codes[labels], (h, w, c))
quantized_image

plt.figure(figsize=(6,6),dpi=200)
plt.imshow(quantized_image)

