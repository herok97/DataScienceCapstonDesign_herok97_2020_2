import cv2, os, random
import numpy as np
from sklearn.model_selection import train_test_split

PATH = '../Data/faces/'
CATEGORIES = ["masked", "no_masked"]

def get_data(size):
    image_size = size
    X = []
    Y = []

    for idex, categorie in enumerate(CATEGORIES):
        label = idex
        image_dir = PATH + categorie + '/'
        print(categorie + '에서 데이터 불러오는 중')
        for top, dir, f in os.walk(image_dir):
            for filename in f:
                print(filename)
                img = cv2.imread(image_dir + filename, cv2.IMREAD_COLOR)
                img = cv2.resize(img, None, fx=image_size / img.shape[1], fy=image_size / img.shape[0])
                X.append(img / 256)
                Y.append(label)
    X = np.array(X)
    Y = np.array(Y)
    random.seed(100)
    return train_test_split(X, Y, test_size=0.1, random_state=random.randint(0, 1000))
