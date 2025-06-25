# Transformer Demo: Mini-Transformer for NLP Task (Next Token Prediction)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ustawienia
vocab_size = 1000     # Rozmiar słownika
embed_dim = 32        # Wymiar osadzenia
seq_len = 10          # Długość sekwencji
num_heads = 2         # Liczba głów atencji
hidden_dim = 64       # Rozmiar warstwy feedforward

# Przykładowe dane wejściowe (indeksy tokenów)
x = torch.randint(0, vocab_size, (32, seq_len))  # batch = 32

def create_targets(x):
    return torch.roll(x, shifts=-1, dims=1)

y = create_targets(x)

# Embedding i pozycja
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

embedding = nn.Embedding(vocab_size, embed_dim)
pos_encoding = PositionalEncoding(embed_dim)

# Transformer Encoder Layer (jedna warstwa)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

# Model końcowy
class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = embedding
        self.pos = pos_encoding
        self.encoder = TransformerBlock(embed_dim, num_heads, hidden_dim)
        self.head = nn.Linear(embed_dim, vocab_size)  # predykcja tokenu

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)
        x = self.encoder(x)
        return self.head(x)

# Trenowanie modelu do predykcji kolejnego tokenu
model = MiniTransformer()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(x)  # (batch, seq_len, vocab_size)
    loss = criterion(output.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Przykład predykcji
model.eval()
x_example = torch.randint(0, vocab_size, (1, seq_len))
with torch.no_grad():
    logits = model(x_example)
    predictions = torch.argmax(logits, dim=-1)

print("Wejście:", x_example)
print("Predykcja następnych tokenów:", predictions)
