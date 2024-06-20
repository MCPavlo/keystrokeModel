import os
import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import KeystrokeDataset, load_data
from model import KeyStrokeClassifier
import torch.nn as nn


def train_model(model, criterion, optimizer, dataloader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for keycode, df, ht, labels in dataloader:
            keycode = keycode.long().to(device)
            df = df.float().to(device)
            ht = ht.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(keycode, df, ht)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")


if __name__ == "__main__":
    # Define paths
    train_path = 'C:\\Users\\pavlo\\PycharmProjects\\pythonProject1\\data\\train'
    model_path = 'C:\\Users\\pavlo\\PycharmProjects\\pythonProject1\\models3'

    # Create directory for saving models if it does not exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Load data
    data = load_data(train_path)

    # Hyperparameters
    key_embed_size = 32
    time_embed_size = 32
    lstm_hidden_size = 64
    attention_size = 128
    batch_size = 32
    learning_rate = 0.001
    epochs = 10

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create and train a model for each user
    for user_id in range(1, 101):
        # Split data into training and validation sets
        user_data = data[data['USER_ID'] == user_id].copy()
        other_data = data[data['USER_ID'] != user_id].sample(len(user_data)).copy()
        user_data.loc[:, 'label'] = 0
        other_data.loc[:, 'label'] = 1
        combined_data = pd.concat([user_data, other_data], ignore_index=True)

        train_data, val_data = train_test_split(combined_data, test_size=0.2, stratify=combined_data['label'])
        train_dataset = KeystrokeDataset(train_data)
        val_dataset = KeystrokeDataset(val_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = KeyStrokeClassifier(key_embed_size, time_embed_size, lstm_hidden_size, attention_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print(f"Training model for user {user_id}")
        train_model(model, criterion, optimizer, train_loader, epochs)

        # Save the model
        model_save_path = os.path.join(model_path, f'user_{user_id}_model.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model for user {user_id} saved to {model_save_path}.\n")
