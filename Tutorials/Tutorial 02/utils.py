import torch
import torch.nn as nn

def train(dataloader, model, n_epochs, optimizer=None, loss_fn=nn.CrossEntropyLoss(), device=torch.device('cpu')):
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), 0.001)
    model.train()
    for epoch in range(n_epochs):
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        print('Epoch {}, loss: {}'.format(epoch+1, loss.item()))
        
def train_mse(dataloader, model, n_epochs, optimizer=None, loss_fn=nn.CrossEntropyLoss(), device=torch.device('cpu')):
    train(dataloader, model, n_epochs, optimizer=None, loss_fn=nn.MSELoss(), device=torch.device('cpu'))