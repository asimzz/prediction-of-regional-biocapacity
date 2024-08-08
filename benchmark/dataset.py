from torch.utils.data import Dataset

class EuroSatDataset(Dataset):
  def __init__(self,images,labels):
    self.images = images
    self.labels = labels

  def __getitem__(self,index):
    return (self.images[index],self.labels[index])

  def __len__(self):
    return len(self.labels)