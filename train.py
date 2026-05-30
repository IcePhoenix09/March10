import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: type[BasicBlock], layers: list[int], num_classes: int = 100) -> None:
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block: type[BasicBlock], out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(num_classes: int = 100) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def evaluate_accuracy(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return correct / total if total > 0 else 0.0

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Augmentation and Loading
    # Standard CIFAR-100 normalization values
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    print("Loading CIFAR-100 dataset...")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    os.makedirs("model_save/checkpoints", exist_ok=True)

    # 3. Model Definition (ResNet-18)
    # Custom implementation for 32x32 CIFAR-100 input
    model = resnet18(num_classes=100)
    model = model.to(device)

    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 5. Training Loop
    # Note: 10 epochs is for demonstration. Real training needs ~100-200.
    epochs = 100
    best_accuracy = 0.0
    print(f"Starting training for {epochs} epochs...")
    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            scheduler.step()

            accuracy = evaluate_accuracy(model, testloader, device)
            best_accuracy = max(best_accuracy, accuracy)
            avg_loss = running_loss / len(trainloader)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                }
                checkpoint_path = f"model_save/checkpoints/checkpoint_epoch_{epoch + 1}.pth"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
    finally:
        final_checkpoint = {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
        }
        torch.save(final_checkpoint, "model_save/checkpoints/checkpoint_latest.pth")
        print("Saved latest checkpoint to model_save/checkpoints/checkpoint_latest.pth")

    # 6. Save Model
    torch.save(model.state_dict(), "resnet18_cifar100.pth")
    print("PyTorch model saved to resnet18_cifar100.pth")

    # 7. Export to ONNX (Task Step 2)
    print("Exporting model to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    torch.onnx.export(
        model, 
        dummy_input, 
        "model.onnx", 
        input_names=['input'], 
        output_names=['output'],
        opset_version=18
    )
    print("Model exported to model.onnx")

if __name__ == "__main__":
    main()
