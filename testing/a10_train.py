import os
import kagglehub
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns

from PIL import Image


CHECKPOINT_DIR = "../model_save/checkpoints/"
MODEL_SAVE_DIR = "../model_save/save_files/"
LOG_DIR = "../model_save/runs/"

translate = {
    'cane': 'dog',
    'cavallo': 'horse',
    'elefante': 'elephant',
    'farfalla': 'butterfly',
    'gallina': 'chicken',
    'gatto': 'cat',
    'mucca': 'cow',
    'pecora': 'sheep',
    'scoiattolo': 'squirrel',
    'ragno': 'spider'
}

def log_confusion_matrix(writer, all_labels, all_predictions, class_names, epoch):
    cm = confusion_matrix(all_labels, all_predictions)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Send the plot to TensorBoard
    writer.add_figure("Confusion Matrix", fig, global_step=epoch)
    plt.close(fig) # Close to save memory

def show_sample_images(images, true_labels, predicted, num_images, class_names):
    fig = plt.figure(figsize=(12, 6))

    for i in range(num_images):
        ax = fig.add_subplot(1, num_images, i + 1, xticks=[], yticks=[])

        img = images[i]
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)

        plt.imshow(img)

        color = "green" if predicted[i] == true_labels[i] else "red"
        ax.set_title(f"P: {class_names[predicted[i]]}\nA: {class_names[true_labels[i]]}", 
                     color=color, fontsize=10)

    plt.show()
    plt.close(fig)


class Net(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        # Standard variables from your original code
        self.model_name = model_name
        self.all_predictions = []
        self.all_labels = []
        self.optimizer = None
        self.criterion = None

        # 1. AlexNet Backbone (Feature Extractor)
        self.backbone = nn.Sequential(
            # Block 1: 11x11 kernel with stride 4 to rapidly downsample the 227x227 image
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Block 2: 5x5 kernel
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Block 3: Three consecutive 3x3 convolutions with NO pooling in between
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Standard practice: forces the output to be exactly 6x6 just in case input size varies
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # 2. AlexNet Classifier (Fully Connected Layers)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5), # Dropout added to prevent overfitting in these massive layers
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, 10), # 10 output classes for your animal dataset
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except the batch dimension
        x = self.fc(x)
        return x
    
    def create_new(self):
        """Create criterion and optimizer for the model"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
    
    def load_model_for_inference(self, file_name):
        save = torch.load(MODEL_SAVE_DIR + file_name, map_location=torch.device('cpu'))
        self.load_state_dict(save)

    def load_checkpoint(self, file_name):
        save = torch.load(CHECKPOINT_DIR + file_name, map_location=torch.device('cpu'))
        self.load_state_dict(save)

    def predict(self, images):
        """ 
        images should be a tensor of shape (N, C, H, W)
        and it should be on gpu
        """
        self.eval()
        with torch.no_grad():
            y_pred = self(images)
            predicted_class = y_pred.argmax(axis=1)

        return predicted_class

    def check_accuracy(self, test_dataloader):
        self.all_predictions = []
        self.all_labels = []

        for x_test, y_test in test_dataloader:
            x_test = x_test.cuda()
            y_test = y_test.cuda()

            predictions = self.predict(x_test)

            self.all_predictions.extend(predictions.cpu().numpy())
            self.all_labels.extend(y_test.cpu().numpy())

        all_predictions = np.array(self.all_predictions)
        all_labels = np.array(self.all_labels)

        correct_results = (all_predictions == all_labels).sum()
        total_samples = len(all_labels)
        accuracy = correct_results / total_samples
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        print(f"Correct results: {correct_results}")
        print(f"Total samples: {total_samples}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")

        return accuracy, f1

    def plot_confusion_matrix(self, class_names=None):
        if not (self.all_labels or self.all_predictions):
            print("[Error] No data inside all_labels or all_predictions")
            return

        # 1. Generate the confusion matrix data
        cm = confusion_matrix(self.all_labels, self.all_predictions)

        # 2. Set up the visual style
        plt.figure(figsize=(10, 8))
        sns.set_theme(style="white")

        # 3. Create the heatmap
        sns.heatmap(
            cm, 
            annot=True,       # Show the numbers in the cells
            fmt='d',          # Use decimal integers
            cmap='Blues',     # Blue color scale
            xticklabels=class_names if class_names else 'auto',
            yticklabels=class_names if class_names else 'auto'
        )

        # 4. Add labels and title
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.show()

    def train_step(self, x, y, optimizer, criterion):
        optimizer.zero_grad()
        y_pred = self(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train_loop(self, train_dataloader, test_dataloader, optimizer, criterion, all_epochs=3, starting_epoch=0):
        EPOCHS = all_epochs - starting_epoch
        if EPOCHS <= 0:
            print("[Error] Training is already finished")
            return

        print("Batch size is - ", BATCH_SIZE)
        print("Number of epochs - ", EPOCHS)

        for epoch in range(EPOCHS):
            self.train(True)

            running_loss = 0.0

            loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

            for batch_idx, (x, y) in enumerate(loader):
                x, y = x.cuda(), y.cuda()
                loss = self.train_step(x, y, optimizer, criterion)
                running_loss += loss

                loader.set_postfix({'loss': f"{loss:.4f}"})

            self.eval()
            accuracy, f1_score = self.check_accuracy(test_dataloader)
            print(f"len(train_dataloader) - {len(train_dataloader)}")
            avg_loss = running_loss / len(train_dataloader)
            print(f"\nEpoch: {epoch + 1} | Loss: {avg_loss} | Accuracy: {accuracy} | F1-score: {f1_score}")
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("accuracy/train", accuracy, epoch)
            writer.add_scalar("F1-score/train", f1_score, epoch)
            log_confusion_matrix(writer, self.all_labels, self.all_predictions, class_names, epoch)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': accuracy,
            }

            file_name = f"{self.model_name}_{epoch}.pth"
            file_path = CHECKPOINT_DIR + file_name
            torch.save(checkpoint, file_path)

        self.save_model()

    def save_model(self):
        torch.save(self.state_dict(), f"{MODEL_SAVE_DIR}{self.model_name}.pth")

    def predict_custom_image(self, image_path, transform, class_names):
            """
            Завантажує власне зображення, обробляє його, повертає та робить передбачення.
            """
            img = Image.open(image_path).convert('RGB')
            

            img = img.rotate(-90, expand=True)
            


            img_tensor = transform(img)
            
            img_tensor = img_tensor.unsqueeze(0).cuda()
            
            self.eval()
            with torch.no_grad():
                output = self(img_tensor)
                predicted_idx = output.argmax(dim=1).item()
                
            predicted_class = class_names[predicted_idx]
            
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"Прогноз моделі: {predicted_class}", fontsize=14, color='blue')
            plt.axis('off')
            plt.show()
            
            print(f"Прогноз моделі: {predicted_class}")
            return predicted_class


if __name__ == '__main__':
    BATCH_SIZE = 32
    MODEL_NAME = "release_v2"

    writer = SummaryWriter(log_dir=LOG_DIR)

    """Data preparation"""
    path = kagglehub.dataset_download("alessiocorrado99/animals10")
    path = path + "/raw-img"

    data_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(), # it converts it to [0, 1] range
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # These numbers calculated on ImageNet
        ])

    dataset = datasets.ImageFolder(root=path, transform=data_transforms)
    class_names = [translate[name] for name in dataset.classes]

    train_data, test_data = random_split(dataset, [0.9, 0.1])
    print(f"Train data len: {len(train_data)}")
    print(f"Test data len: {len(test_data)}")

    train_dataloader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True
        )

    test_dataloader = DataLoader(
        test_data, 
        batch_size=BATCH_SIZE,
        num_workers=4, 
        pin_memory=True
        )
    print(f"Size of image tensor: {train_dataloader.dataset[0][0].size()}")

    """Model preparation"""
    model = Net(MODEL_NAME).cuda()
    
    # Commented out to prevent a crash if the file doesn't exist yet
    model.load_model_for_inference("release_v2.pth")

    # Loop through images in ../img and predict their classes
    for img_file in os.listdir("../img"):
        img_path = os.path.join("../img", img_file)
        model.predict_custom_image(img_path, data_transforms, class_names)

    """Training execution (Uncomment to train)"""
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()
    # model.train_loop(train_dataloader, test_dataloader, optimizer, criterion, all_epochs=10, starting_epoch=0)

    # model.check_accuracy(test_dataloader)
    # model.plot_confusion_matrix(class_names)
    
    # Grab a batch to properly supply data to the visualizer
    test_images, test_labels = next(iter(test_dataloader))
    test_preds = model.predict(test_images.cuda()).cpu().numpy()
    show_sample_images(test_images.numpy(), test_labels.numpy(), test_preds, 10, class_names)

    writer.flush()
    writer.close()