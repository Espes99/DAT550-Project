import torch.nn as nn
import torch
from utils.embedding_loader import EmbeddingLoader

class RNNClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim, embedding, num_layers=2, dropout=0.3, rnn_type="lstm", bidirectional=False, attention_layer="None", rtn_attn_weight=True, batch_first=False, num_heads=4):
        super(RNNClassifier, self).__init__()

        self.embedding = embedding
        self.emb_dim = self.embedding.embedding_dim
        self.lstm_bi = bidirectional
        self.return_attn_weights = rtn_attn_weight
        self.batch_first = batch_first
        self.hidden_dim = hidden_dim * 2 if self.lstm_bi else hidden_dim

        # RNN model
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(self.emb_dim, hidden_dim, num_layers=num_layers, batch_first=batch_first, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(self.emb_dim, hidden_dim, num_layers=num_layers, batch_first=batch_first, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError("rnn_type must be either 'lstm' or 'gru'")
        
        # Attention Layer definition
        self.use_attention = attention_layer.lower() != "none"
        if attention_layer.lower() == "custom":
            self.attn_layer = CustomAttentionLayer(hidden_dim * 2 if self.lstm_bi else hidden_dim)
        elif attention_layer.lower() == "custom_dot":
            self.attn_layer = CustomDotProductAttention()
        elif attention_layer.lower() == "custom_mlp":
            self.attn_layer = CustomMLPAttention(hidden_dim * 2 if self.lstm_bi else hidden_dim)
        elif attention_layer.lower() == "mha":
            self.attn_layer = PredefinedMultiheadAttention(hidden_dim * 2 if self.lstm_bi else hidden_dim, num_heads=num_heads)
        elif self.use_attention:
            raise ValueError(f"Unknown attention_layer type: {attention_layer}")

        self.fc = nn.Linear(hidden_dim * 2 if self.lstm_bi else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded)

        # For LSTM, hidden is a tuple (hidden_state, cell_state)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        
        if self.use_attention:
            if isinstance(self.attn_layer, CustomDotProductAttention):
                final_hidden = hidden[-1]
                if self.return_attn_weights:
                    context, attn_weights = self.attn_layer(output, final_hidden, return_weights=True)
                else:
                    context = self.attn_layer(output, final_hidden)
            else:
                if self.return_attn_weights:
                    context, attn_weights = self.attn_layer(output, return_weights=True)
                else:
                    context = self.attn_layer(output)
            return self.fc(context) if not self.return_attn_weights else (self.fc(context), attn_weights)
        else:
            final_hidden = hidden[-1]
            return self.fc(final_hidden)

class CustomAttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_outputs, return_weights=False):
        weights = torch.softmax(self.attn(rnn_outputs), dim=1)
        context = (rnn_outputs * weights).sum(dim=1)

        if return_weights:
            return context, weights.squeeze(-1)
        else:
            return context
        
class CustomDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rnn_outputs, final_hidden, return_weights=False):
        scores = torch.bmm(rnn_outputs, final_hidden.unsqueeze(2)).squeeze(2)
        weights = torch.softmax(scores, dim=1).unsqueeze(2)
        context = (rnn_outputs * weights).sum(dim=1)

        if return_weights:
            return context, weights.squeeze(-1)
        else:
            return context
        
class CustomMLPAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, rnn_outputs, return_weights=False):
        weights = torch.softmax(self.attn(rnn_outputs), dim=1)
        context = (rnn_outputs * weights).sum(dim=1)

        if return_weights:
            return context, weights.squeeze(-1)
        else:
            return context
        
class PredefinedMultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, rnn_outputs, return_weights=False):
        attn_output, attn_weights = self.attn(rnn_outputs, rnn_outputs, rnn_outputs)

        context = attn_output.mean(dim=1)

        if return_weights:
            return context, attn_weights.mean(dim=1)
        else:
            return context