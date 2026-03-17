# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset

This experiment demonstrates transfer learning using a pre-trained ResNet18 model on a custom image dataset. Instead of training a deep neural network from scratch, the pre-trained model’s feature extraction layers are reused, and only the final classification layer is retrained. This approach reduces training time, requires less data, and achieves high accuracy.

## DESIGN STEPS
### STEP 1:
Data Preprocessing – Resize all images to 224×224 and convert them into tensors suitable for ResNet input.

### STEP 2: 
Dataset Loading – Organize images into train/test sets and load them using ImageFolder and DataLoader.

### STEP 3:
Load Pretrained Model – Use ResNet18 trained on ImageNet as the base model.

### STEP 4:
Modify Final Layer – Freeze earlier layers and replace the fully connected layer to match the number of dataset classes.

### STEP 5:
Train and Evaluate – Train only the final layer, then test the model and analyze results using a confusion matrix and classification report.

## PROGRAM
```python
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# Modify the final fully connected layer to match the dataset classes
for param in model.parameters():
    param.requires_grad = False   # freeze earlier layers
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

```
```python
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: RIHAN AHAMED S")
    print("Register Number:  212224040276")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="1132" height="787" alt="image" src="https://github.com/user-attachments/assets/794e3a57-6a5e-4722-9ae6-419514ba30dd" />
<img width="301" height="67" alt="image" src="https://github.com/user-attachments/assets/bcb7d20d-9859-4e86-8d61-e2a68f0fa48b" />




### Confusion Matrix
<img width="767" height="632" alt="image" src="https://github.com/user-attachments/assets/9c8007a3-35d5-4e89-8e85-f685680280c7" />


### Classification Report

<img width="681" height="243" alt="image" src="https://github.com/user-attachments/assets/295904fe-39fc-4971-896a-25dc17854baa" />



### New Sample Prediction
<img width="781" height="558" alt="image" src="https://github.com/user-attachments/assets/d67af3d3-aa8d-4c27-95d1-1508691b5ed1" />
<img width="879" height="568" alt="image" src="https://github.com/user-attachments/assets/dea4e1cd-43bb-42bf-86bd-62715881a5ee" />




## RESULT
The Implementation of Transfer Learning for classification using VGG-19 architecture is successful.
