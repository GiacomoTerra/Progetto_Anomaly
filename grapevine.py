#import packages
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.utils import make_grid

class GrapeDataset(Dataset):
	# inizializza il Dataset object con la directory e la grandezza delle patch
	def __init__(self, data_dir, patch_size):
		self.data_dir = data_dir
		self.image_files = self.load_image_files()
		self.patch_size = patch_size
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
		width, height = image.size
		patches = []
		for i in range(0, height, self.patch_size):
			for j in range(0, width, self.patch_size):
				patch = image.crop((j, i , j+self.patch_size, i+self.patch_size))
				tensor_patch = ToTensor()(patch)
				patches.append(tensor_patch)
		return patches
		
	def __len__(self):
		return len(self.image_files)

writer = SummaryWriter()
data_dir = "test_dataset"
patch_size = 200
dataset = GrapeDataset(data_dir, patch_size)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size = 8, shuffle = True)
for i, data in enumerate(dataset):
	images = data
	images_tensor = torch.stack(images)
	transform = transforms.Compose([transforms.Normalize(mean=[0.5, 0,5, 0.5], std=[0.5,0.5,0.5])])
	transformed_images = transform(images_tensor)
	image_grid = make_grid(images, nrow=4, normalize=True, scale_each=True)
	writer.add_image('Images', image_grid, i)
writer.close()
