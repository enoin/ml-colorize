import math

import torch
from torch import nn
from nn.dataset import get_data_loaders, get_test_data_loaders, get_test_images
from nn.network import ColorizerNN
from util.device import get_device
from util.visualize import show_test_images


class ColorizeNNRunner:
    MODEL_NAME = 'model.pth'
    learning_rate = 0.001
    batch_size = 4
    epochs = 4

    def __init__(self, test_data_path, train_data_path):
        self._train_data_path = train_data_path
        self._test_data_path = test_data_path
        self._device = get_device()

    def test_loop(self, dataloader, model, loss_fn):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self._device), y.to(self._device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    def train_loop(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        stat_on_every = math.floor(size / (dataloader.batch_size * 100))
        running_loss = 0
        for batch, (X, y) in enumerate(dataloader, 0):
            X, y = X.to(self._device), y.to(self._device)

            predicted = model(X)

            optimizer.zero_grad()
            loss = loss_fn(predicted, y)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch % stat_on_every == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"batch: [{batch}] loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return running_loss

    def train(self):

        model = ColorizerNN()
        model = model.to(self._device)
        model.train()
        print(model)

        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        color_loss = nn.MSELoss()

        loader = get_data_loaders(self._train_data_path, self.batch_size, 8, 50000)
        test_dataloader = get_test_data_loaders(self._test_data_path)

        for t in range(self.epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop(loader, model, color_loss, optimizer)
            self.test_loop(test_dataloader, model, color_loss)

        print("Done!")
        # ----------
        torch.save(model, ColorizeNNRunner.MODEL_NAME)

    def load_model(self):
        model = torch.load(ColorizeNNRunner.MODEL_NAME, weights_only=False)
        model = model.to(self._device)
        model.eval()
        return model

    def test(self):
        model = self.load_model()
        images = get_test_images(self._test_data_path, 7)
        for image_state in images:
            input_tensor = image_state.gray.to(self._device).unsqueeze(1)
            colorized = model(input_tensor).squeeze(0).cpu().detach()
            image_state.colorized = colorized

        print(images[0].ground_truth[0])
        print(images[0].colorized)
        show_test_images(images)

        print(model)
