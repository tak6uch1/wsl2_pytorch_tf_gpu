import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import nn

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
            nn.Softmax(dim=1),
        )

        # weight init
        for m in self.layers.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.layers(x)

if __name__ == '__main__':

    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Devide: ", end='')
    print(device)

    # Model definition
    model = cnn().to(device)
    opt = torch.optim.Adam(model.parameters())

    # Load data
    bs = 128 # batch size                                                                  
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True)

    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False)

    # Training                                                                             
    print('train')
    model = model.train()
    for iepoch in range(12):
        for iiter, (x, y) in enumerate(trainloader, 0):

            x = x.to(device)
            y = torch.eye(10)[y].to(device)

            y_ = model.forward(x) # y_.shape = (bs, 84)                                    

            # loss: cross-entropy                                                          
            eps = 1e-7
            loss = -torch.mean(y*torch.log(y_+eps))

            opt.zero_grad()
            loss.backward()
            opt.step()

            if iiter%100==0:
                print('%03d epoch, %05d, loss=%.5f' %
                      (iepoch, iiter, loss.item()))

    # Test                                                                                 
    print('test')
    total, tp = 0, 0
    model = model.eval()
    for (x, label) in testloader:

        x = x.to(device)

        y_ = model.forward(x)
        label_ = y_.argmax(1).to('cpu')

        total += label.shape[0]
        tp += (label_==label).sum().item()

    acc = tp/total
    print('test accuracy = %.3f' % acc)

