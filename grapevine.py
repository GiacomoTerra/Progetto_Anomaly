#import packages
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

class GrapeDataset(Dataset):
	# inizializza il Dataset object con la directory e la grandezza delle patch
	def __init__(self, data_dir, transform=None):
		self.data_dir = data_dir
		self.image_files = self.load_image_files()
		self.transform = transform
	# carica su un vettore i path dei file immagini
	def load_image_files(self):
		image_files = []
		for root, dirs, files in os.walk(self.data_dir):
			for file in files:
				if file.endswith(".jpg") or file.endswith(".png"):
					image_path = os.path.join(root, file)
					image_files.append(image_path)
		return image_files
		
	def __getitem__(self, index):
		image_path = self.image_files[index]
		image = Image.open(image_path)
		if self.transform is not None:
			image = self.transform(image)
		return image
		
	def __len__(self):
		return len(self.image_files)

# Apro un Summary Writer su TensorBoard
writer = SummaryWriter()

# Definizione delle trasformazioni
transform = transforms.Compose([
	# Ridimensiona l'immagine alle dimensioni desiderate
	transforms.Resize((300, 300)), 
	# Converte l'immagine in un tensore
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Creazione del dataset
data_dir = "test_dataset"
dataset = GrapeDataset(data_dir, transform=transform)
print(len(dataset))

# Creazione del dataloader
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

# Caricamento di un batch di immagini dal dataloader
dataiter = iter(dataloader)
images = next(dataiter)

# creazione di una griglia di immagini per la visualizzazione
img_grid = make_grid(images, nrow=4, normalize=True, scale_each=True)

# Aggiunta della griglia di immagini a TensorBoard
writer.add_images('Sample Images', img_grid)

# Chiusura del SummaryWriter
writer.close()
