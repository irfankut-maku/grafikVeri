import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import VisualQA

# Hyperparametreler
num_epochs = 10
learning_rate = 0.001
batch_size = 16
num_classes = 10  # Sınıf sayısını ihtiyacınıza göre ayarlayın

# Modeli, kayıp fonksiyonunu ve optimizasyonu tanımlayın
model = VisualQA(num_classes=num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Dataset ve DataLoader'ı tanımlayın
dataset = CustomDataset(json_file='data/dataset.json', img_dir='data/images', transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Cihazı belirleyin
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Eğitim döngüsü
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, questions, answers in dataloader:
        images = images.to(device)

        # Soruları ve cevapları uygun tensörlere dönüştürün
        # Örneğin: questions ve answers için vektörleştirme yapılması gerekebilir

        # Optimizasyonu ve kaybı sıfırlayın
        optimizer.zero_grad()

        # Modelden tahmin alın
        outputs = model(images, questions)

        # Kayıp hesaplayın
        loss = criterion(outputs, answers)
        loss.backward()
        optimizer.step()

        # Kayıp değerini güncelleyin
        running_loss += loss.item()

    # Epoch başına ortalama kayıp değeri
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Model eğitimi tamamlandı ve kaydedildi!")
torch.save(model.state_dict(), "model/model.pth")
