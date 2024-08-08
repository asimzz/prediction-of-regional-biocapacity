from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import image_slicer
from image_slicer import join
import imageio

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_list = []
        self.prediction = {}
        self.pred_index = {}
        self.labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                       "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
        self.indices = {i: [] for i in range(len(self.labels))}
        self.label_counts = {label: 0 for label in self.labels}
        self.tiles = []

    
    def slice_image(self, num_tiles):
        self.tiles = image_slicer.slice(self.image_path, num_tiles)
    
    def read_tiles(self):
        self.image_list = [imageio.imread(tile.filename) for tile in self.tiles]
    
    def classify_tiles(self, classifier):
        for i, img in enumerate(self.image_list):
            img_resized = np.resize(img, (64, 64, 3))
            img_array = image.img_to_array(img_resized)
            img_expanded = np.expand_dims(img_array, axis=0)
            pred = classifier.predict(img_expanded)[0]
            
            self.prediction[i] = pred
            index_max = np.argmax(pred)
            self.pred_index[i] = index_max
            
            self.indices[index_max].append(i)

    def count_labels(self):
        self.label_counts = {label: sum(idx == i for idx in self.pred_index.values()) for i, label in enumerate(self.labels)}
    
    def calculate_biocapacity(self, area):
        YF = {
            "Crop_Land": 0.437252,
            "Forest_Land": 0.439149,
            "Grazing_Land": 1,
            "Marine_Fishing": 1.47469,
            "Infrastructure": 0.437252,
            "Inland_Fishing": 1
        }
        EF = {
            "Crop_Land": 2.49939691398113,
            "Forest_Land": 1.26215878768974,
            "Grazing_Land": 0.453121058342895,
            "Marine_Fishing": 0.364490579331025,
            "Infrastructure": 2.49939691398113,
            "Inland_Fishing": 0.364490579331025
        }
        temp = {
            "Crop_Land": self.label_counts['AnnualCrop'] + self.label_counts['PermanentCrop'],
            "Forest_Land": self.label_counts['Forest'],
            "Grazing_Land": self.label_counts['Pasture'] + self.label_counts['HerbaceousVegetation'],
            "Marine_Fishing": self.label_counts['River'],
            "Infrastructure": self.label_counts['Industrial'] + self.label_counts['Highway'] + self.label_counts['Residential'],
            "Inland_Fishing": self.label_counts['SeaLake']
        }

        bio = 0
        for land_type in EF:
            percentage = temp[land_type] / self.num_tiles
            land_area = percentage * area
            bio += land_area * YF[land_type] * EF[land_type]

        return bio / area


    def process(self):
        self.read_tiles()
        self.classify_tiles()
        self.count_labels()
        print(self.label_counts)
        self.plot_tiles(4, 48, 64, 4, 4)  # Example usage for plotting Industrial tiles

if __name__ == "__main__":
    image_path = '/content/drive/MyDrive/Graduation Project/totilast.jpg'
    num_tiles = 2500
    area = 620
    classifier = ...  # Load or define your classifier here

    processor = ImageProcessor(image_path, classifier)
    processor.slice_image(num_tiles)
    processor.process()
    bio_capacity = processor.calculate_biocapacity(area)
    print(bio_capacity)
