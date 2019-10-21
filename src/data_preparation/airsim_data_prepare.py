import glob, cv2
import numpy as np

imgs_1 = glob.glob(dirs[i]+'/images/*.png')
imgs_1 = sorted(imgs_1, key=lambda x: x.split('.')[0].split('/')[-1].split('_')[-1])

def stack_images(i1, i2, i3):
    a = cv2.imread(i1)
    b = cv2.imread(i2)
    c = cv2.imread(i3)
    return np.hstack([a, b, c])

for i in range(len(imgs_1)-4):
    z = stack_images(imgs_1[i], imgs_1[i+2], imgs_1[i+4])
    cv2.imwrite('v2/{:05d}.png'.format(i), z)

count = 0
for k in range(len(dirs)):
    imgs_1 = glob.glob(dirs[k]+'/images/*.png')
    imgs_1 = sorted(imgs_1, key=lambda x: x.split('.')[0].split('/')[-1].split('_')[-1])

    for i in range(count, count+len(imgs_1)-4):
        z = stack_images(imgs_1[i-count], imgs_1[i-count+2], imgs_1[i-count+4])
        cv2.imwrite('v2/{:05d}.png'.format(i), z)

    count += len(imgs_1) - 4
