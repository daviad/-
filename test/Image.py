import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img=mpimg.imread('/Users/dxw/Downloads/stinkbug.png')
# print(img)
# imgplot = plt.imshow(img)
# print(imgplot)
lum_img = img[:,:,0]
# print(lum_img)
# imgplot = plt.imshow(lum_img)
plt.hist(lum_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
plt.show()
