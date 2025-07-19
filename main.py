#dependecies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#datset
text = """
To be or not to be that is the question Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune Or to take arms against a sea of troubles
And by opposing end them To die to sleep No more and by a sleep to say we end
The heartache and the thousand natural shocks That flesh is heir to tis a consummation
Devoutly to be wished To die to sleep To sleep perchance to dream ay there's the rub
For in that sleep of death what dreams may come When we have shuffled off this mortal coil
Must give us pause there's the respect That makes calamity of so long life
"""

# Preprocessing
tokens = text.lower().split()
vocab = sorted(set(tokens))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

# Create training data
context_size = 3
embed_size = 64
hidden_size = 128

def make_data(tokens, context_size):
    data = []
    for i in range(context_size, len(tokens)):
        context = [word2idx[tokens[j]] for j in range(i - context_size, i)]
        target = word2idx[tokens[i]]
        data.append((context, target))
    return data

data = make_data(tokens, context_size)

class NPLMDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)

loader = DataLoader(NPLMDataset(data), batch_size=16, shuffle=True)

# NPLM model
class NPLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.hid = nn.Linear(embed_size * context_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
    def forward(self, x):
        x = self.emb(x).view(x.size(0), -1)
        h = torch.tanh(self.hid(x))
        return F.log_softmax(self.out(h), dim=1)

# Train model
model = NPLM()
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Training model...")
for epoch in range(15):
    total = 0
    for context, target in loader:
        optimizer.zero_grad()
        out = model(context)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch+1} Loss: {total:.4f}")

# Generate sentence
def generate_sentence(seed, n=10):
    model.eval()
    words = seed.lower().split()
    context = words[-context_size:]
    for _ in range(n):
        context_ids = torch.tensor([[word2idx.get(w, 0) for w in context]])
        with torch.no_grad():
            output = model(context_ids)
        next_idx = torch.multinomial(torch.exp(output[0]), 1).item()
        next_word = idx2word[next_idx]
        words.append(next_word)
        context = words[-context_size:]
    return ' '.join(words)

# --- 🔁 User input loop ---
print("\nGenerate sentences from your prompt!")
while True:
    seed = input("\nEnter a seed (3+ words, e.g., 'to be or'): ").strip()
    if not seed or len(seed.split()) < context_size:
        print("Please enter at least 3 words.")
        continue
    sentence = generate_sentence(seed, n=15)
    print("Generated:", sentence)