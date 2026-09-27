import torch
import torch.nn as nn

with open("text.txt", "r", encoding="utf-8") as file:
    text = file.read()

text = text.lower()
words = text.split()

word_to_id = {}
for word in words:
    if word not in word_to_id:
        word_to_id[word] = len(word_to_id)

id_to_word = {value: key for key, value in word_to_id.items()}
vocab_size = len(word_to_id)
encoded_words = [word_to_id[w] for w in words]

sequence_length = 3
X = []
y = []
for i in range(len(encoded_words) - sequence_length):
    X.append(encoded_words[i:i + sequence_length])
    y.append(encoded_words[i + sequence_length])

X_train = torch.tensor(X, dtype=torch.long)
y_train = torch.tensor(y, dtype=torch.long)

class TextModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.linear = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

model = TextModel(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 150

for epoch in range(epochs):
    optimizer.zero_grad()

    output = model(X_train)
    loss = criterion(output, y_train)

    loss.backward()
    optimizer.step()

    print(f"Эпоха {epoch + 1}, ошибка: {loss.item():.4f}")


user_phrase = input("\nВведите начальную фразу (из 3 слов): ").lower().split()
input_ids = [word_to_id[w] for w in user_phrase if w in word_to_id]
input_sequence = torch.tensor([input_ids], dtype=torch.long)

output = model(input_sequence)

predicted_id = torch.argmax(output, dim=1).item()
predicted_word = id_to_word[predicted_id]

print(f"\nПредсказанное следующее слово: {predicted_word}")
print(f"Результат: {' '.join(user_phrase)} -> {predicted_word}")


num_generate = int(input("\nСколько слов сгенерировать дальше? "))
generated_tokens = list(user_phrase)

for _ in range(num_generate):
    current_input = [word_to_id[w] for w in generated_tokens[-sequence_length:]]
    input_seq = torch.tensor([current_input], dtype=torch.long)
    
    out = model(input_seq)
    next_id = torch.argmax(out, dim=1).item()
    next_word = id_to_word[next_id]
    
    generated_tokens.append(next_word)

print("\nИтоговый сгенерированный текст:")
print(" ".join(generated_tokens))