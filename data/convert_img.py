import cv2
import os

dd = 'data/00-test-0'
od = 'data/00-test'

for i, file in enumerate(os.listdir(dd)):
    if file.lower().endswith(('jpeg', 'jpg', 'png', 'webp')):
        path = os.path.join(dd, file)
        npath = os.path.join(od, f'{i:06}.jpg')
        img = cv2.imread(path)
        if img is None:
            print('can not open ', path)
            continue
        cv2.imwrite(npath, img)
        print(f'convert {path} -> {npath}')