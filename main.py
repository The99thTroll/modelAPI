from numpy.random import multinomial
import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps

x, y = fetch_openml("mnist_784", version=1, return_X_y=True)
x_test, x_train, y_test, y_train = train_test_split(x, y, train_size=7500, test_size=2500, random_state=9)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(x_train_scaled, y_train)

def getPrediction(images):
    im_pil = Image.open(images)
    img_bw = im_pil.convert("L")
    img_bw_resize = img_bw.resize((28, 28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(img_bw_resize, pixel_filter)
    
    img_bw_resize_inv_scale = np.clip(img_bw_resize - min_pixel, 0, 255)
    max_pixel = np.max(img_bw_resize)
    img_bw_resize_inv_scale = np.asarray(img_bw_resize_inv_scale)/max_pixel
    
    test_sample = np.array(img_bw_resize_inv_scale).reshape(1,784)
    test_pred = clf.predict(test_sample)
    
    return test_pred[0]