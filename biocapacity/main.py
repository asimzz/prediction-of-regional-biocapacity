"""
## RGB Model and Biocapacity Calculations


```
for more informations : endomamoro9@gmail.com, asimabdalla99@gmail.com
```

### Initialization
"""

import numpy as np
from PIL import Image
from model import EuroSatRGBClassifier
from preprocessing import load_dataset, split_dataset



# Download and read RGB EuroSAT images from URL 
# RGB file URL
url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"

images, labels = load_dataset(url)

num_classes = len(np.array(np.unique(labels)))

X_train, X_test, y_train, y_test = split_dataset(images,labels,num_classes)





model_file = '/content/drive/MyDrive/Graduation Project/ds-rgb/M8.h5'


classifier = EuroSatRGBClassifier(X_train.shape[1:],num_classes,model_file)

# history = classifier.train(X_train, X_test, y_train, y_test,epochs=20,batch_size=64)
# score = classifier.model.evaluate(X_test, y_test, verbose=0)
classifier.load_weights(model_file)

# Predicting Khartoum State Labels




from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
# %matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
import image_slicer
from image_slicer import join
from PIL import Image
import imageio

#importing the full image
img=Image.open('/content/drive/MyDrive/Graduation Project/totilast.jpg')
#specifying the number of tiles
num_tiles = 2500
#slicing the image
tiles = image_slicer.slice('/content/drive/MyDrive/Graduation Project/totilast.jpg', num_tiles)
#adding the tiles into image list
image_list = []
for tile in tiles:
    img = imageio.imread(tile.filename)
    image_list.append(img)
# work on the copy only
list1 = image_list
#making a prediction dict to save the predictions of each tile
prediction = dict.fromkeys(range(0,2499 ))
pred_index = {}
#loop through the tiles
AnnualCropindex = []
Forestindex = []
HerbaceousVegetationindex = []
Highwayindex = []
Industrialindex = []
Pastureindex = []
PermanentCropindex = []
Residentialindex = []
Riverindex = []
SeaLakeindex = []
for i in range(0,2499):
  #resizing so it can fit the model
  imgx = np.resize(list1[i],(64,64,3))
  img1 = image.img_to_array(imgx)
  img = np.expand_dims(img1, axis = 0)
  x =classifier.predict(img)
  prediction[i] = x[0]
  index_max = np.argmax(prediction[i])
  pred_index[i] = index_max
  newlist = Image.fromarray(image_list[i])
  tiles[i].image = newlist
  if(pred_index[i] == 0):
    AnnualCropindex.append(i)
  if(pred_index[i] == 1):
    Forestindex.append(i)
  if(pred_index[i] == 2):
    HerbaceousVegetationindex.append(i)
  if(pred_index[i] == 3):
    Highwayindex.append(i)
  if(pred_index[i] == 4):
    Industrialindex.append(i)
  if(pred_index[i] == 5):
    Pastureindex.append(i)
  if(pred_index[i] == 6):
    PermanentCropindex.append(i)
  if(pred_index[i] == 7):
    Residentialindex.append(i)
  if(pred_index[i] == 8):
    Riverindex.append(i)
  if(pred_index[i] == 9):
    SeaLakeindex.append(i)


imgf = join(tiles)

plt.imshow(imgf)

#labels
labels = {
  "AnnualCrop":0,
  "Forest" : 0,
    "HerbaceousVegetation" :0,
    "Highway":0,
    "Industrial":0,
    "Pasture":0,
    "PermanentCrop":0,
    "Residential":0,
    "River":0,
    "SeaLake":0
}
#the number of times a label found in the image
labels["AnnualCrop"] = sum(value == 0 for value in pred_index.values())
labels["Forest"] = sum(value == 1 for value in pred_index.values())
labels["HerbaceousVegetation"] = sum(value == 2 for value in pred_index.values())
labels["Highway"] = sum(value == 3 for value in pred_index.values())
labels["Industrial"] = sum(value == 4 for value in pred_index.values())
labels["Pasture"] = sum(value == 5 for value in pred_index.values())
labels["PermanentCrop"] = sum(value == 6 for value in pred_index.values())
labels["Residential"] = sum(value == 7 for value in pred_index.values())
labels["River"] = sum(value == 8 for value in pred_index.values())
labels["SeaLake"] = sum(value == 9 for value in pred_index.values())
labels


indeustrialindex2 = Industrialindex[48:64]
fig = plt.figure(figsize=(10, 10))
rows = 4
columns = 4
q = 1
for i in indeustrialindex2:
  fig.add_subplot(rows, columns, q)
  q = q + 1
  plt.imshow(tiles[i].image)

"""# Tuti Island Biocapacity Calculation"""

#BioCapacity = sum(area_of_khartoum * Yeild_factor_of_Sudan * Equivalince_factor)
Area = 620 #for 2.94 zoom in google earth
YF = {
    "Crop_Land":0.437252,
    "Forest_Land":0.439149,
    "Grazing_Land":1,
    "Marine_Fishing":1.47469,
    "Infrastructure":0.437252,
    "Inland_Fishing":1
}
EF = {
    "Crop_Land":2.49939691398113,
    "Forest_Land":1.26215878768974,
    "Grazing_Land":0.453121058342895,
    "Marine_Fishing":0.364490579331025,
    "Infrastructure":2.49939691398113,
    "Inland_Fishing":0.364490579331025
}
temp = {
    "Crop_Land":0,
    "Forest_Land":0,
    "Grazing_Land":0,
    "Marine_Fishing":0,
    "Infrastructure":0,
    "Inland_Fishing":0
}

temp['Crop_Land'] = labels['AnnualCrop'] + labels['PermanentCrop']
temp['Forest_Land'] = labels['Forest']
temp['Grazing_Land'] = labels['Pasture'] + labels['HerbaceousVegetation']
temp['Marine_Fishing'] = labels['River']
temp['Infrastructure'] = labels['Industrial'] + labels['Highway'] + labels['Residential']
temp['Inland_Fishing'] = labels['SeaLake']
bio = 0
for i in EF:
    print(i,temp[i])
    percentage = temp[i] / num_tiles
    area = percentage * Area
    print(percentage,area)
    bio += area * YF[i] * EF[i]
    print(bio)
print(bio/Area)