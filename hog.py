import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure
from PIL import Image
from PIL import ImageEnhance
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import feature
import numpy as np



#image = color.rgb2gray(data.astronaut())
image = Image.open('original_images/indoor/257971_2008934073530_6552451_o.jpg').resize((512, 512), Image.ANTIALIAS).convert('L')
image = sobel(image)
#image = feature.canny(np.array(image))
#enh = ImageEnhance.Sharpness(image)
#image = enh.enhance(3)

fd, hog_image = hog(image, orientations=20, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True, normalise=True)

print fd
print len(fd)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

print len(hog_image)

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()
