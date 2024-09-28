import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader


class GoogleNetV3Classifier:
    def __init__(self, dataset_dir, batch_size=32, learning_rate=0.001, num_epochs=10, device=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.train_loader, self.val_loader, self.test_loader = self.load_data(dataset_dir)

        self.model = self.build_model()
        self.model = self.model.to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_data(self, dataset_dir):
        # AiArtData为0, RealArtData为1
        dataset = datasets.ImageFolder(root=dataset_dir, transform=self.transform)

        train_size = int(0.6 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def build_model(self):
        # 加载预训练的Inception v3并修改最后一层
        model = models.inception_v3(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()  # 二分类输出
        )
        return model

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device).float()

            outputs = self.model(images)
            outputs = outputs.view(-1)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        print(f'Train Loss: {avg_loss:.4f}')

    def test(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float().view(-1, 1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                predicted = (outputs >= 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Test Loss: {test_loss / len(self.test_loader)}, Test Accuracy: {accuracy:.2f}%")

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float().view(-1, 1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(self.val_loader)}")

    def train(self):
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            self.train_one_epoch()
            self.validate()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f'Model loaded from {path}')
