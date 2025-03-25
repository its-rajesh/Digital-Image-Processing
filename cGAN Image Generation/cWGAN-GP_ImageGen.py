import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
num_classes = 10
img_channels = 1
embed_dim = 50
lambda_gp = 10
batch_size = 64
num_epochs = 100
n_critic = 5
lr = 0.0002
beta1 = 0.5
beta2 = 0.9

# Create directories
sample_dir = 'samples'
os.makedirs(sample_dir, exist_ok=True)

# Generator Definition
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 7, 7)),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        embedding = self.label_embed(labels)
        combined = torch.cat([noise, embedding], dim=1)
        return self.model(combined)

# Critic Definition
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.label_projection = nn.Linear(embed_dim, 128 * 7 * 7)
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, img, labels):
        out = self.conv(img)
        embed = self.label_embed(labels)
        projected = self.label_projection(embed).view(out.shape)
        out = out + projected
        return self.fc(out.view(out.size(0), -1))

# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Initialize models
generator = Generator().to(device)
critic = Critic().to(device)
generator.apply(weights_init)
critic.apply(weights_init)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Training loop
for epoch in range(num_epochs):
    for i, (real_imgs, real_labels) in enumerate(dataloader):
        real_imgs, real_labels = real_imgs.to(device), real_labels.to(device)
        
        # Train Critic
        optimizer_C.zero_grad()
        
        # Generate fake images
        z = torch.randn(real_imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z, real_labels)
        
        # Compute validity and gradient penalty
        real_validity = critic(real_imgs, real_labels)
        fake_validity = critic(fake_imgs.detach(), real_labels)
        
        # Gradient penalty
        epsilon = torch.rand(real_imgs.size(0), 1, 1, 1).to(device)
        interpolated = epsilon * real_imgs + (1 - epsilon) * fake_imgs.detach()
        interpolated.requires_grad_(True)
        validity_interpolated = critic(interpolated, real_labels)
        
        gradients = torch.autograd.grad(
            outputs=validity_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(validity_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        # Critic loss
        critic_loss = (-(real_validity.mean() - fake_validity.mean()) 
                       + lambda_gp * gradient_penalty)
        critic_loss.backward()
        optimizer_C.step()
        
        # Train Generator every n_critic steps
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            
            # Generate fake images
            gen_labels = torch.randint(0, num_classes, (real_imgs.size(0),)).to(device)
            gen_imgs = generator(z, gen_labels)
            
            # Generator loss
            gen_loss = -critic(gen_imgs, gen_labels).mean()
            gen_loss.backward()
            optimizer_G.step()
            
            # Print progress
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} "
                      f"Critic Loss: {critic_loss.item():.4f} Gen Loss: {gen_loss.item():.4f}")
        
        # Save samples
        if i % 100 == 0:
            with torch.no_grad():
                fixed_z = torch.randn(10, latent_dim).to(device)
                fixed_labels = torch.arange(0, 10).to(device)
                samples = generator(fixed_z, fixed_labels).cpu()
                save_image(samples, os.path.join(sample_dir, f"epoch{epoch}_batch{i}.png"),
                           nrow=10, normalize=True)
    
    # Save models
    torch.save(generator.state_dict(), f'generator_epoch{epoch}.pth')
    torch.save(critic.state_dict(), f'critic_epoch{epoch}.pth')

# Inference code example
def generate_samples(model_path, num_samples=10):
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        labels = torch.arange(0, num_samples).to(device)
        samples = generator(z, labels).cpu()
        save_image(samples, 'final_samples.png', nrow=10, normalize=True)

# Usage
generate_samples('generator_epoch99.pth')