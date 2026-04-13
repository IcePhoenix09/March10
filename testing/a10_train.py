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


# test_loader = iter(dataloader)
# images, labels = next(test_loader)

# plt.imshow(images[0].permute(1, 2, 0)) 
# plt.title(f"Label: {labels[0]}")
# plt.show()

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

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, (5, 5), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(4, 4, (5, 5), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(4, 8, (5, 5), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, (5, 5), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Linear(1600, 10)

        self.model_name = model_name

        self.all_predictions = []
        self.all_labels = []


    def forward(self, x):
        # net = x[:, None].type(torch.float32)
        net = self.backbone(x)
        net = net.view(-1, 16 * 10 * 10)
        net = self.fc(net)
        # net = nn.functional.relu(net)
        return net


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


def train_step(x, y, model, optimizer, criterion):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_loop(model, train_dataloader, test_dataloader, optimizer, criterion, all_epochs=3, starting_epoch=0):

    EPOCHS =  all_epochs - starting_epoch
    if EPOCHS <= 0:
        print("[Error] Training is already finished")
        return 

    print("Batch size is - ", BATCH_SIZE)
    print("Number of epochs - ", EPOCHS)

    for epoch in range(EPOCHS):
        model.train(True)

        running_loss = 0.0

        loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            loss = train_step(x, y, model, optimizer, criterion)
            running_loss += loss

            loader.set_postfix({'loss': f"{loss:.4f}"})

        model.eval()
        accuracy, f1_score = model.check_accuracy(test_dataloader)
        print(f"len(train_dataloader) - {len(train_dataloader)}")
        avg_loss = running_loss / len(train_dataloader)
        print(f"\nEpoch: {epoch + 1} | Loss: {avg_loss} | Accuracy: {accuracy} | F1-score: {f1_score}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("accuracy/train", accuracy, epoch)
        writer.add_scalar("F1-score/train", f1_score, epoch)
        log_confusion_matrix(writer, model.all_labels, model.all_predictions, class_names, epoch)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': accuracy,
        }

        # timestamp = time.time()
        file_name = f"{model.model_name}_{epoch}.pth"
        file_path = CHECKPOINT_DIR + file_name
        torch.save(checkpoint, file_path)
    
    save_model(model)

def save_model(model):
    torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}{model.model_name}.pth")

def load_model(model, file_name):
    save = torch.load(MODEL_SAVE_DIR + file_name, map_location=torch.device('cpu'))
    model.load_state_dict(save)

def load_checkpoint(model, file_name):
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint = torch.load(CHECKPOINT_DIR + file_name)

    # Restore states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return optimizer, epoch


if __name__ == '__main__':
    BATCH_SIZE = 32
    MODEL_NAME = "test_v2"

    # writer = SummaryWriter(log_dir=LOG_DIR)
    writer = SummaryWriter()

    """Data preparetion"""
    path = kagglehub.dataset_download("alessiocorrado99/animals10")
    path = path + "/raw-img"

    data_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(), # it converte it to [0, 1] range
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # This numbers calculated on ImageNet
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
    # train_loader = iter(dataloader)

    test_dataloader = DataLoader(
        test_data, 
        batch_size=BATCH_SIZE,
        num_workers=4, 
        pin_memory=True
        )
    print(f"Size of image tensor: {train_dataloader.dataset[0][0].size()}")

    """Model preparetion"""
    model = Net(MODEL_NAME).cuda()
    load_model(model, "test_v2.pth")
    # optimizer, epoch = load_checkpoint(model, 'checkpoint_epoch_2.pth')


    # test_item = next(iter(train_dataloader))[0].cuda()
    # print(model.forward(test_item).size())

    # test_item = next(iter(train_dataloader))[0].cuda()
    # print(model.forward(test_item).size())
    # print(model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()

    # train_loop(model, train_dataloader, test_dataloader, optimizer, criterion, all_epochs=1, starting_epoch=0)

    model.check_accuracy(test_dataloader)

    model.plot_confusion_matrix(class_names)
    log_confusion_matrix(writer, model.all_labels, model.all_predictions, class_names, epoch=0)
    show_sample_images(model.all_predictions, 6, class_names)
    # log_sample_images(model, test_dataloader, 6, writer)

    writer.flush()
    writer.close()

    # print(test_data.dataset.tensors[0][test_data.indices].size())
