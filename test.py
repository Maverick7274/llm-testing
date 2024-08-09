import torch
import torch.nn.functional as F
import torch.optim as optim
import math

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scaled Dot-Product Attention
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).item()

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

# Multi-Head Attention
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)
        self.out = torch.nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.q_linear(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        concat = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.num_heads)
        output = self.out(concat)
        return output, attn_weights

# Positional Encoding
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Feed-Forward Network
class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Encoder Layer
class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output, _ = self.multi_head_attention(x, x, x, mask)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x

# Decoder Layer
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.masked_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.layer_norm3 = torch.nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        masked_attn_output, _ = self.masked_attention(x, x, x, tgt_mask)
        x = self.layer_norm1(x + masked_attn_output)
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.layer_norm2(x + cross_attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm3(x + ff_output)
        return x

# Full Transformer Model
class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6):
        super(Transformer, self).__init__()
        self.encoder_embedding = torch.nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = torch.nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.final_linear = torch.nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        src = self.encoder_embedding(src)
        src = self.positional_encoding(src)
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        return src

    def decode(self, tgt, memory, src_mask, tgt_mask):
        tgt = self.decoder_embedding(tgt)
        tgt = self.positional_encoding(tgt)
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, src_mask, tgt_mask)
        return tgt

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        output = self.final_linear(output)
        return output

# Greedy Decoding
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(device)
    for i in range(max_len - 1):
        out = model.decode(ys, memory, src_mask, tgt_mask=None)
        prob = model.final_linear(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word).to(device)], dim=1)
        if next_word == end_symbol:
            break
    return ys

# Example hyperparameters
src_vocab_size = 10000
tgt_vocab_size = 10000
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
num_epochs = 1000
batch_size = 64
src_seq_len = 20
tgt_seq_len = 20
start_symbol = 1
end_symbol = 2

# Instantiate the model and move to device
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len)).to(device)
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    src_mask = None
    tgt_mask = None
    outputs = model(src, tgt_input, src_mask, tgt_mask)
    loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Inference
model.eval()
src = torch.randint(0, src_vocab_size, (1, src_seq_len)).to(device)
src_mask = None
max_len = 10
decoded_sequence = greedy_decode(model, src, src_mask, max_len, start_symbol)
print(f'Decoded sequence: {decoded_sequence}')
