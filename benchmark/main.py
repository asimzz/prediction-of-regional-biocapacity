import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn import metrics
from tqdm import tqdm
from dataset import EuroSatDataset
from helpers import load_dataset_from_pickle, split_dataset, tiff_loader
from device import get_default_device, to_device, DeviceDataLoader
from models import ResNet152, GoogleNet, EfficientNet
from trainer import ModelTrainer
from metrics import check_accuracy, check_metric



"""# Data Preprocessing"""

data_origin = "/path/of/the/dataset"


'''
NOTE:
Since the dataset is expensive to load everytime using the ImageFolder we saved the
an instance from the EuroSatDataset class as a pickle file for cheaper loadness, but
you can load the data for the first time by uncommenting the code below

You download the dataset from the Eurosat page: https://github.com/phelber/eurosat
'''


# image_dataset = ImageFolder(root=data_origin, loader=tiff_loader)
# images = []
# labels = []


# image_dataset = EuroSatDataset(images, labels)

# for image, label in tqdm(image_dataset):
#     images.append(image)
#     labels.append(labels)


save_path = "/path/to/save/the/pickle/file"

image_dataset =  load_dataset_from_pickle(save_path)



"""## Split The Dataset"""
train_data, val_data, test_data = split_dataset(image_dataset)

device = get_default_device()
batch_size = 64
kw = {"num_workers": 8, "pin_memory": True} if device == "cuda" else {}


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kw)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, **kw)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, **kw)

train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

"""#Model Archetictures"""

resnet_model = ResNet152(13, 10)
googleNet_model = GoogleNet(13, 10)
enet_model = EfficientNet(
    version="b0",
    num_classes=10,
)


"""# Training The ResNet model"""

resnet_model = to_device(resnet_model, device)
resnet_trainer = ModelTrainer(resnet_model, "resnet", train_loader, val_loader)
history = [resnet_trainer.evaluate()]


# history += resnet_trainer.train(epochs=20,save_path="",)

# The path where you saved the model weights
saved_model = torch.load(save_path)

saved_resnet_model = to_device(ResNet152(13, 10), device)
saved_resnet_model.load_state_dict(saved_model["resnet"])

check_accuracy(train_loader, resnet_model)

"""# Training The GoogleNet model"""

googleNet_model = to_device(googleNet_model, device)
googleTrainer = ModelTrainer(googleNet_model, "resnet", train_loader, val_loader)
history = [googleTrainer.evaluate()]


# history += googleTrainer.train(epochs=20,save_path="")

saved_model = torch.load(save_path)

saved_googleNet_model = to_device(GoogleNet(13, 10), device)
saved_googleNet_model.load_state_dict(saved_model["googleNet_model"])

check_accuracy(train_loader, googleNet_model)

"""# Training The EffNet model"""

efficentnet_model = to_device(enet_model, device)
efficentnet_trainer = ModelTrainer(
    resnet_model, "efficentnet", train_loader, val_loader
)
history = [efficentnet_trainer.evaluate()]

# history += efficentnet_trainer.train(epochs=20,save_path="")


saved_model = torch.load(save_path)

saved_effNet_model = to_device(EfficientNet("b0", 10), device)
saved_effNet_model.load_state_dict(saved_model["effNet_model"])

check_accuracy(train_loader, saved_effNet_model)

"""##Precision, Recall and F1 Score For GoogleNet"""

check_metric(test_loader, googleNet_model, metrics.recall_score, "macro", "Recall")

check_metric(
    test_loader, googleNet_model, metrics.precision_score, "macro", "Precision"
)

check_metric(test_loader, googleNet_model, metrics.f1_score, "macro", "F1")

"""# Precision, Recall and F1 score for ResNet152"""

check_metric(test_loader, saved_resnet_model, metrics.recall_score, "macro", "Recall")

check_metric(
    test_loader, saved_resnet_model, metrics.precision_score, "macro", "Precision"
)

check_metric(test_loader, saved_resnet_model, metrics.f1_score, "macro", "F1")
