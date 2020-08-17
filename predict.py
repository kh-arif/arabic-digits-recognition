import numpy as np
from keras.models import model_from_json
from PIL import Image

def rot_digit(digit):
    return np.fliplr(np.rot90(digit, axes=(1,0)))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
im = Image.open("tux.bmp")
im = im.resize((28,28))
p = np.array(im)
p = rgb2gray(p)
p = p /255.0
p = rot_digit(p)
p = p.reshape(1, 28, 28,1)
preds = loaded_model.predict(p)[0]
fst = np.argmax(preds)
fst_p = preds[fst]
preds = np.delete(preds, fst)
snd = np.argmax(preds)
snd_p = preds[snd]
preds = np.delete(preds, snd)
thd = np.argmax(preds)
thd_p = preds[thd]
print("{0} {1:.0%} {2} {3:.0%} {4} {5:.0%}".format(fst, fst_p, snd, snd_p, thd, thd_p))
