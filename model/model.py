import torch
import torch.nn as nn
import torchvision.models as models


class VisualQA(nn.Module):
    def __init__(self, num_classes, vocab_size=10000, embed_dim=300):
        super(VisualQA, self).__init__()
        # Görseller için önceden eğitilmiş bir CNN (ör. ResNet18)
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)

        # Soru için Embedding ve LSTM katmanı
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=256, num_layers=1, batch_first=True)

        # Sınıflandırıcı
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, images, questions):
        # Görseller için CNN özelliklerini çıkar
        img_features = self.cnn(images)

        # Sorular için Embedding ve LSTM özelliklerini çıkar
        embedded = self.embedding(questions)
        _, (q_features, _) = self.lstm(embedded)
        q_features = q_features[-1]

        # Görsel ve soru özelliklerini birleştir
        combined = torch.cat((img_features, q_features), dim=1)

        # Sınıflandırıcıya gönder
        output = self.fc(combined)
        return output
