#import packages
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from grapevine import *
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
Image.LOAD_TRUNCATED_IMAGES = True

def matplotlib_imshow(img, one_channel=False):
	if one_channel:
		img = img.mean(dim=0)
	img = img / 2 + 0.5
	npimg = img.numpy()
	if one_channel:
		plt.imshow(npimg, cmap="Greys")
	else:
		plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Definizione delle trasformazioni
data_transform = transforms.Compose([
	transforms.Resize((256, 256)),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

class ConvAutoencoder(nn.Module):
	def __init__(self):
		super(ConvAutoencoder, self).__init__()
		# Encoder
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=9, stride=1, padding=4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(16,18, kernel_size=9, stride=2, padding=4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(18,20, kernel_size=9, stride=2, padding=4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=3),
			nn.Conv2d(20,22, kernel_size=9, stride=3, padding=4),
			nn.LeakyReLU(0.2, inplace=True)
		)
		# Decoder
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(22,20, kernel_size=9, stride=3, padding=4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(20,18, kernel_size=9, stride=2, padding=4, output_padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(18,16, kernel_size=9, stride=2, padding=4, output_padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ConvTranspose2d(16,3, kernel_size=9, stride=1, padding=4),
			nn.Sigmoid()
		)
	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
# Creazione dell'istanza del modello
model = ConvAutoencoder()
print(model)
# Definizione dell'ottimizzatore
optimizer = optim.Adam(model.parameters(), lr=0.5e-4)
# Definizione della funzione di loss
criterion = nn.MSELoss()

writer = SummaryWriter('runs/esempio')
train_dir = "test_dataset"
test_dir = "/percorso/della/directory"

train_dataset = GrapeDataset(train_dir, data_transform)
train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
#test_dataset = GrapeDataset(test_dir, patch_size)
#test_dataloader = DataLoader(test_dataset, batch_size = 8, shuffle = False)
print(len(train_dataset))
#print(len(test_dataset))


# Numero di immagini casuali da selezionare
num_sample_images = 4
# Selezionare alcune immagini casuali dal dataloader di training
sample_images = []
for i, data in enumerate(train_dataloader):
	sample_images.extend(data)
	if i == num_sample_images - 1:
		break
# Crea una grid di immagini
img_grid = torchvision.utils.make_grid(sample_images, nrow=num_sample_images, normalize=True, scale_each=True)
matplotlib_imshow(img_grid, one_channel = True)
writer.add_image('Sample Images', img_grid)
# Aggiungo il grafico del modello al writer
sample_batch = next(iter(train_dataloader))
writer.add_graph(model, img_grid)


# Numero di epoche
num_epochs = 28

# Switcho il modello in train mode
model.train()

# Addestramento del modello
for epoch in range(num_epochs):
	total_loss = 0.0
	# Iterazione attraverso i dati di addestramento
	for i, data in enumerate(train_dataloader, 0):
		inputs = data
		inputs = Variable(inputs)
		# Azzaramento dei gradienti
		optimizer.zero_grad()
		# Calcolo dell'output del modello
		outputs = model(inputs)
		# Calcolo della loss
		loss = criterion(outputs, inputs)
		total_loss += loss.item()
		# Calcolo dei gradienti e aggiornamento dei pesi
		loss.backward()
		optimizer.step()
		# Stampa della loss per ogni 750 passi
		if (i+1)%750 == 0:
			printf(f'Epoch: {epoch+1}/{num_epochs}, Step: {i+1}/{len(train_dataloader)}, Loss: {total_loss/750}')
			# Salva la loss su tensorboard
			step = epoch * len(train_dataloader) + (i+1)
			writer.add_scalar('Loss', total_loss/750, step)
			total_loss = 0.0
			# Aggiunta dei dati dell'encoder al projector
			encoder_data = []
			encoder_labels = []
			# Iterazione attraverso il dataset di addestramento
			for j, data in enumerate(train_dataloader, 0):
				inputs = data
				inputs = Variable(inputs)
				# Calcolo dell'output dell'encoder
				encoded = model.encoder(inputs)
				encoder_data.append(encoded)
				encoder_labels.append(f'Image_{j}')
			# Concatenazione dei dati dell'encoder
			encoder_data = torch.cat(encoder_data, dim=0)
			encoder_labels = torch.tensor(labels)
			# Aggiunta dei dati dell'encoder al projector
			writer.add_embedding(encoder_data, metadata=labels, global_step = step)
# Chiusura dell'oggetto SummaryWriter
writer.close()
print("Finished Training!\n")


# Valutazione del modello sul set di test
#total_test_loss = 0.0
# Cambio in modalità validazione
#model.eval()

# Validazione del modello
#for i, data in enumerate(test_dataloader, 0):
	#inputs = data
	#inputs = Variable(inputs)
	# Calcolo dell'output del modello
	#outputs = model(inputs)
	# Calcolo della loss
	#loss = criterion(outputs, inputs)
	#total_test_loss += loss.item()
#avg_test_loss = total_test_loss / len(test_dataloader)
#print(f'Test Loss: {avg_test_loss}\n')
#print("Finished Validation!")

