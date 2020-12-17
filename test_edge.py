"""
 * Python script to demonstrate Canny edge detection.
 *
 * usage: python CannyEdge.py <filename> <sigma> <low_threshold> <high_threshold>
"""
import skimage
import skimage.feature
import skimage.viewer
import sys

# read command-line arguments
filename = "21.jpg"
sigma = 100
low_threshold = 100
high_threshold = 100

# load and display original image as grayscale
image = skimage.io.imread(fname=filename, as_gray=True)
viewera = skimage.viewer(image=image)
viewera.show()

edges = skimage.feature.canny(
    image=image,
    sigma=sigma,
    low_threshold=low_threshold,
    high_threshold=high_threshold,
)
# display edges
viewer = skimage.viewer.ImageViewer(edges)
viewer.show()