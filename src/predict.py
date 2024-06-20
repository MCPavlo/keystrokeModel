import os
import torch
import pandas as pd
from model import KeyStrokeClassifier

def load_data(path):
    data = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            user_data = pd.read_csv(os.path.join(path, file))
            user_id = int(file.split('_')[0])
            user_data['USER_ID'] = user_id
            user_data['KEYCODE'] = user_data['KEYCODE'].astype(int)
            user_data['df'] = user_data['df'].astype(float)
            user_data['ht'] = user_data['ht'].astype(float)
            data.append(user_data)
    return pd.concat(data, ignore_index=True)

def prepare_tensors(user_data, sequence_length):
    user_data = user_data.iloc[:sequence_length]
    keycode = torch.tensor(user_data['KEYCODE'].values).long().view(1, sequence_length)
    df = torch.tensor(user_data['df'].values).float().view(1, sequence_length, 1)
    ht = torch.tensor(user_data['ht'].values).float().view(1, sequence_length, 1)
    return keycode, df, ht

def test_model(model, keycode, df, ht):
    model.eval()
    with torch.no_grad():
        outputs = model(keycode, df, ht)
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()

if __name__ == "__main__":
    # Define paths
    test_path = 'C:\\Users\\pavlo\\PycharmProjects\\pythonProject1\\data\\test'
    model_path = 'C:\\Users\\pavlo\\PycharmProjects\\pythonProject1\\models'

    # Load test data
    test_data = load_data(test_path)

    # Define parameters
    key_embed_size = 32
    time_embed_size = 32
    lstm_hidden_size = 64
    attention_size = 128
    sequence_length = 200

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize results and error counter
    results = []
    false_negative_counter = 0

    # Initialize counters for each classifier
    classifier_correct_counts = {model_id: 0 for model_id in range(1, 101)}
    classifier_incorrect_counts = {model_id: 0 for model_id in range(1, 101)}
    classifier_false_negative_counts = {model_id: 0 for model_id in range(1, 101)}

    total_users_tested = 0

    for user_id in range(1, 101):
        print(f'\nTesting models for user {user_id} data:')
        user_test_data = test_data[test_data['USER_ID'] == user_id].copy()

        if len(user_test_data) < sequence_length:
            print(f'Not enough data for user {user_id}. Skipping.')
            continue

        keycode, df, ht = prepare_tensors(user_test_data, sequence_length)
        total_users_tested += 1

        for model_id in range(1, 101):
            model = KeyStrokeClassifier(key_embed_size, time_embed_size, lstm_hidden_size, attention_size).to(device)
            model_load_path = os.path.join(model_path, f'user_{model_id}_model.pth')
            model.load_state_dict(torch.load(model_load_path))

            prediction = test_model(model, keycode.to(device), df.to(device), ht.to(device))

            # Check if the model correctly identifies the user
            if model_id == user_id:
                if prediction[0] == 1:
                    false_negative_counter += 1
                    classifier_false_negative_counts[model_id] += 1
            is_correct = (prediction[0] == 0) if (model_id == user_id) else (prediction[0] == 1)
            results.append((user_id, model_id, is_correct))

            # Update counters for each classifier
            if is_correct:
                classifier_correct_counts[model_id] += 1
            else:
                classifier_incorrect_counts[model_id] += 1

    # Calculate accuracy
    correct_predictions = sum(1 for result in results if result[2])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f'\nOverall accuracy: {accuracy:.2f}')
    print(f'Total false negatives for target users: {false_negative_counter}')
    print(f'Total users tested: {total_users_tested}')