import torch
from torch import nn

def train_step(model : torch.nn.Module,
              dataloader : torch.utils.data.DataLoader,
              loss_func : torch.nn.Module,
              optimizer : torch.optim.Optimizer,
              device : torch.device):

  model.train()

  train_loss,train_acc = 0,0

  for batch,(X,y) in enumerate(dataloader):
    X,y = X.to(device),y.to(device)


    y_pred = model(X)

    loss = loss_func(y_pred,y)

    train_loss += loss.item()

    y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)

    train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss,train_acc


def test_step(model : torch.nn.Module,
              dataloader : torch.utils.data.DataLoader,
              loss_func : torch.nn.Module,
              optimizer : torch.optim.Optimizer,
              device : torch.device):

  model.eval()

  with torch.inference_mode():
    test_loss,test_acc = 0,0

    for batch,(X,y) in enumerate(dataloader):
      X,y = X.to(device),y.to(device)

      y_pred = model(X)

      loss = loss_func(y_pred,y)

      test_loss += loss.item()

      y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)

      test_acc += (y_pred_class == y).sum().item() / len(y_pred)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

  return test_loss,test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_func=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_func=loss_fn,
                                        optimizer=optimizer,
                                        device=device)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results at the end of the epochs
    return results