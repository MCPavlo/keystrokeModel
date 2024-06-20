import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyStrokeClassifier(nn.Module):
    def __init__(self, key_embed_size, time_embed_size, lstm_hidden_size, attention_size):
        super(KeyStrokeClassifier, self).__init__()
        self.key_embed = nn.Embedding(num_embeddings=256, embedding_dim=key_embed_size)
        self.df_embed = nn.Linear(1, time_embed_size)
        self.ht_embed = nn.Linear(1, time_embed_size)
        self.lstm = nn.LSTM(input_size=key_embed_size + 2 * time_embed_size, hidden_size=lstm_hidden_size,
                            num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)
        self.attention = nn.Linear(lstm_hidden_size * 2, attention_size)
        self.attention_combine = nn.Linear(attention_size, 1)
        self.fc = nn.Linear(lstm_hidden_size * 2, 2)

    def forward(self, keycode, df, ht):
        key_embed = self.key_embed(keycode)
        df_embed = torch.relu(self.df_embed(df.unsqueeze(-1)))
        ht_embed = torch.relu(self.ht_embed(ht.unsqueeze(-1)))

        # Add sequence dimension
        df_embed = df_embed.unsqueeze(1)
        ht_embed = ht_embed.unsqueeze(1)

        # Ensure correct dimensions for concatenation
        key_embed = key_embed.unsqueeze(1) if len(key_embed.shape) == 2 else key_embed

        combined = torch.cat((key_embed, df_embed, ht_embed), dim=2)
        lstm_out, _ = self.lstm(combined)

        attention_weights = torch.tanh(self.attention(lstm_out))
        attention_weights = F.softmax(self.attention_combine(attention_weights), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        output = self.fc(context_vector)
        return output
