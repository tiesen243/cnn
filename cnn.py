import torch

from tqdm import tqdm
from torch import nn, optim
from torchsummary import summary
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.history = []
        self.layers = nn.ModuleList(layers)
        self.to(device)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return x

    def summary(self, input_shape, batch_size):
        summary(self, input_shape, batch_size)

    def config(self, loss: nn.Module, optimizer: optim.Optimizer):
        if not isinstance(loss, nn.Module):
            raise TypeError("loss must be a torch.nn.Module instance")
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer instance")

        self.criterion = loss
        self.optimizer = optimizer

    def fit(self, train_loader: DataLoader, epochs: int = 10, verbose: bool = True):
        for epoch in range(epochs):
            self.train()

            # Split the train set into train and validation set
            train_set, val_set = random_split(train_loader.dataset, [50000, 10000])
            train_set = DataLoader(train_set, batch_size=64, shuffle=True)
            val_set = DataLoader(val_set, batch_size=64, shuffle=True)

            loss_list = []

            for images, labels in tqdm(train_set, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)

                self.optimizer.zero_grad()

                outputs = self(images)
                loss = self.criterion(outputs, labels)
                loss_list.append(loss.item())

                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.eval()

                total = 0
                accuracy = 0
                val_loss = []

                for images, labels in val_set:
                    images, labels = images.to(device), labels.to(device)

                    outputs = self(images)
                    total += labels.size(0)
                    predicted = torch.argmax(outputs, dim=1)

                    accuracy += (predicted == labels).sum().item()
                    val_loss.append(self.criterion(outputs, labels).item())

                # Calculate the mean loss and accuracy
                mean_val_loss = sum(val_loss) / len(val_loss)
                mean_val_acc = 100 * (accuracy / total)
                loss = sum(loss_list) / len(loss_list)
                self.history.append((loss, mean_val_loss, mean_val_acc))

                if verbose:
                    print(
                        f"Loss: {loss:.4f}, Val Loss: {mean_val_loss:.4f}, Val Accuracy: {mean_val_acc:.2f}%"
                    )

        return self.history

    def predict(self, x):
        predicted = []

        with torch.no_grad():
            self.eval()
            for images, _ in x:
                images = images.to(device)

                outputs = self(images)
                predicted.append(torch.argmax(outputs, 1))

        return predicted
