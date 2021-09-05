import torch
from torchvision import transforms, datasets
from InceptionNet_v4 import InceptionNet

from utils import train, valid

# instantiate InceptionNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = InceptionNet().to(device)



# create training, validation and test datasets
# augument data (whatever augmentations faces are susceptible to)

batch_size = 32
target_size = (256,256)
train_dir = ''
valid_dir = ''


data_transform = transforms.Compose([
    transforms.RandomSizedCrop(target_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#############################################################################################################################################



# set necessary parameters for training
learning_rate = 0.001
loss_fn = torch.nn.TripletMarginLoss()
optimizer = torch.optim.Adamax(net.parameters(), lr=learning_rate)
epochs = 50

# training loop
for e in range(epochs):
    train(dataloader=train_loader, net=net, loss_fn=loss_fn, optimizer=optimizer, device=device)
    print("Epoch",e+1)
    valid(dataloader=train_loader, net=net, loss_fn=loss_fn, optimizer=optimizer, device=device)