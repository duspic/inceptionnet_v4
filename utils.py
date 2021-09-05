import torch
def train(dataloader, net, loss_fn, optimizer, device):

    size  = len(dataloader.dataset)

    for batch, (x,y) in enumerate(dataloader):

        x,y = x.to(device), y.to(device)

        pred = net(x)
        loss = loss_fn(pred,y)

    # 3-step backpropagation process

        # reset gradients to zero
        optimizer.zero_grad()

        # calculate new gradients backwards
        loss.backward()

        # set parameters according to new gradients
        optimizer.step()

        loss, current = loss.item(), batch*len(X)
        print("["+str(current)+"/"+str(size)+']'+'loss:'+str(loss))



def valid(dataloader, net, loss_fn, device):

    size = len(dataloader.dataset)

    # prepare net for evaluation
    net.eval()
    test_loss, correct = 0.0, 0.0

    # similar loop as train, but without backpropagation
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device), y.to(device)
            pred = net(x)
            test_loss += loss_fn(pred, y).item()
            # compares the predicted and true class for the entire batch and sums the correctly predicted ones
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= size
    correct /= size
    print("Accuracy: "+str(100*correct) + "%, Average loss: "+str(100*test_loss))