import numpy as np
import torch
import matplotlib.pyplot as plt

from lib.model import SimpleNN

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

#Seed random number generation for reproducible results
torch.manual_seed(1)

#All parameters here
max_epochs = 8
learning_rate = 1e-3
width = 16 #neurons in hidden layers
num_affine_layers = 10 
noise_amplitude = 1e-2 #random noise added to our data


#Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts the images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])
# Download the training and test datasets
train_dataset = MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='data', train=False, download=True, transform=transform)

# Print dataset sizes
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

i = torch.unsqueeze( torch.arange( 28 ), 0 )
j = torch.unsqueeze( torch.arange( 28 ), 1 )

#Generate checkerboard masks
mask1 = (i + j) % 2
mask2 = 1 - mask1

mask1 = mask1.flatten()
mask2 = mask2.flatten()

net = SimpleNN( num_affine_layers, width, mask1, mask2 )
net = torch.load( 'network.pth' )

optimizer = torch.optim.Adam( net.parameters(), lr = learning_rate)

# Create a DataLoader for the training dataset
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,  # Number of samples per batch
    shuffle=True,   # Shuffle the data at every epoch
)

for epoch in np.arange( max_epochs ):
    train_loss = 0.0

    for images, labels in train_loader:
        # `images` is a batch of images, `labels` is a batch of corresponding labels
        #print(images.shape)  # Shape of the batch of images
        images = torch.flatten(images, 1, 3) #flatten down to (b, 28*28)
        images = images + noise_amplitude * torch.randn( images.shape )
        #print(images.shape)  # Shape of the batch of images
        
        # map data to latent space z
        z, logdet = net.inverse( images )

        # calculate the negative loglikelihood
        loss = torch.log( z.new_tensor([2*np.pi])) + torch.mean(torch.sum(0.5*z**2, -1) - logdet)
        train_loss = train_loss + loss.item()

        #Compute gradients and step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if( epoch % 1 == 0):
        print( f"Epoch {epoch}: loss = {train_loss}")

torch.save(net, "network.pth")

#Sample the latent space with a normal distribution
num_images = 100
z = torch.normal( mean=0, std=1, size=(num_images,28*28) )

with torch.no_grad():
    generated_digits, _ = net.forward(z)

for i in np.arange(num_images):
    plt.figure()
    image = torch.reshape( generated_digits[i,:], (28,28) )
    #plt.scatter( traj_artifical[:,0], traj_artifical[:,1], s=marker_size )
    plt.imshow(image)
    plt.colorbar
    plt.clim([-0.5, 0.5])
    plt.title('Generative AI digit!')
    plt.savefig( f"figures/{i}.png" )
    plt.close()