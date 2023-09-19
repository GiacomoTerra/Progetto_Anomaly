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
	transforms.ToTensor(),
	transforms.Resize((320, 320), antialias = True),
])

def double_conv(in_channels, out_channels):
	conv = nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace = True),
		nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace = True),
	)
	return conv

def crop_img(tensor, target_tensor):
	target_size = target_tensor.size()[2]
	tensor_size = tensor.size()[2]
	delta = tensor_size - target_size
	delta = delta // 2
	return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]
	
class UNet(nn.Module):
	def __init__(self):
		super (UNet, self).__init__()
		self.max_pool_2x2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
		self.down_conv_1 = double_conv(3, 64)
		self.down_conv_2 = double_conv(64, 128)
		self.down_conv_3 = double_conv(128, 256)
		self.down_conv_4 = double_conv(256, 512)
		self.down_conv_5 = double_conv(512, 1024)
		
		self.up_trans_1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)
		self.up_conv_1 = double_conv(1024, 512)
		self.up_trans_2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)
		self.up_conv_2 = double_conv(512, 256)
		self.up_trans_3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)
		self.up_conv_3 = double_conv(256, 128)
		self.up_trans_4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
		self.up_conv_4 = double_conv(128, 64)
		self.out = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 1)
	
	def forward(self, image):
		# encoder
		x1 = self.down_conv_1(image)
		x2 = self.max_pool_2x2(x1)
		x3 = self.down_conv_2(x2)
		x4 = self.max_pool_2x2(x3)
		x5 = self.down_conv_3(x4)
		x6 = self.max_pool_2x2(x5)
		x7 = self.down_conv_4(x6)
		x8 = self.max_pool_2x2(x7)
		x9 = self.down_conv_5(x8)
		
		#decoder
		x = self.up_trans_1(x9)
		y = crop_img(x7, x)
		x = self.up_conv_1(torch.cat([x, y], 1))
		x = self.up_trans_2(x)
		y = crop_img(x5, x)
		x = self.up_conv_2(torch.cat([x, y], 1))
		x = self.up_trans_3(x)
		y = crop_img(x3, x)
		x = self.up_conv_3(torch.cat([x, y], 1))
		x = self.up_trans_4(x)
		y = crop_img(x1, x)
		x = self.up_conv_4(torch.cat([x, y], 1))
		
		x = self.out(x)
		return x
	
# Creazione dell'istanza del modello
model = UNet()
print(model)
# Definizione dell'ottimizzatore
optimizer = optim.Adam(model.parameters(), lr=0.5e-4)
# Definizione della funzione di loss
criterion = nn.MSELoss()

writer = SummaryWriter('runs/esempio')
train_dir = "train_dataset"
test_dir = "test_dataset"

train_dataset = GrapeDataset(train_dir, data_transform)
train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
test_dataset = GrapeDataset(test_dir, data_transform)
test_dataloader = DataLoader(test_dataset, batch_size = 8, shuffle = False)
print(len(train_dataset))
print(len(test_dataset))

# get some random training images
dataiter = iter(train_dataloader)
images = next(dataiter)
# creazione di una griglia di immagini per la visualizzazione
img_grid = make_grid(images, normalize=True, scale_each=True)

# Show images
matplotlib_imshow(img_grid, one_channel=True)

# Aggiunta della griglia di immagini a TensorBoard
writer.add_images('Train Images', img_grid.unsqueeze(0))

# Aggiunta del grafico su tensorboard
writer.add_graph(model, images)

# Numero di epoche
num_epochs = 10

# Switcho il modello in train mode
model.train()
print("Start Training\n")

# Ciclo di addestramento
for epoch in range(num_epochs):
	running_loss = 0.0
	for batch in train_dataloader:
		inputs = batch
		inputs = Variable(inputs)
		# Azzaramento dei gradienti
		optimizer.zero_grad()
		# Calcolo dell'output del modello
		outputs = model(inputs)
		# Calcolo della loss
		loss = criterion(outputs, inputs)
		# Calcolo dei gradienti e aggiornamento dei pesi
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
	# Calcola la perdita media per epoca
	epoch_loss = running_loss / len(train_dataloader)
	# Stampa la perdita media
	print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}')
	# Registra la perdita su TensorBoard
	writer.add_scalar('Loss/Train', epoch_loss, epoch)
	
	# Registra anche le immagini in input e le loro ricostruzioni
	for i, batch in enumerate(train_dataloader):
		if i == 0: # Registra solo la prima coppia di immagini 
			inputs = batch
			inputs = Variable(inputs)
			# Calcolo dell'output del modello
			outputs = model(inputs)
			# Registra le coppie di immagini in input e ricostruite su TensorBoard
			writer.add_image(f"Train/Input_{epoch}", inputs[0], epoch)
			writer.add_image(f"Train/Reconstruction_{epoch}", outputs[0], epoch)

print("Finished Training!\n")

# Cambio in modalità validazione
model.eval()

# Valutazione del modello sul set di test
total_test_loss = 0.0
# Validazione del modello
for data in enumerate(test_dataloader, 0):
	inputs = data
	# Calcolo dell'output del modello
	outputs = model(inputs)
	# Registra le coppie di immagini in input e ricostruite su TensorBoard
	writer.add_image("Test/Input", inputs[0], epoch)
	writer.add_image("Test/Reconstruction", outputs[0], epoch)
	# Calcolo della loss
	loss = criterion(outputs, inputs)
	total_test_loss += loss.item()
avg_test_loss = total_test_loss / len(test_dataloader)
# Registra la loss di validazione su TensorBoard
writer.add_scalar("Loss/Validation", avg_test_loss, epoch)
print(f'Test Loss: {avg_test_loss:.4f}\n')
print("Finished Validation!")

# Chiusura dell'oggetto SummaryWriter
writer.close()
