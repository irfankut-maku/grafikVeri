import torch
from PIL import Image
from model.model import VisualQA  # Model dosyasından model sınıfını içe aktar
from dataset import CustomDataset  # Veriyi yüklemek için dataset sınıfını içe aktar
from torchvision import transforms
import json

# Modeli yükle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisualQA(num_classes=100).to(device)
model.load_state_dict(torch.load("model/model.pth"))
model.eval()  # Modeli değerlendirme moduna al


# Görseli ve soruyu işle ve tahmin al
def predict(image_path, question_text):
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Batch boyutu 1 olacak şekilde

    # Soruyu tensor haline getirme
    question = torch.tensor([ord(char) for char in question_text], dtype=torch.long).unsqueeze(0).to(device)

    # Tahmin yapma
    with torch.no_grad():
        output = model(image, question)
        _, predicted = torch.max(output, 1)
    return predicted.item()


# Test veri kümesini yükle
with open('data/test_dataset.json', 'r', encoding='utf-8') as file:
    test_data = json.load(file)

# Test işlemi
correct_predictions = 0
total_predictions = 0

for entry in test_data:
    image_path = entry["image"]
    for qa_pair in entry["qa_pairs"]:
        question = qa_pair["question"]
        true_answer = qa_pair["answer"]

        # Tahmini al
        predicted_answer = predict(image_path, question)

        # Tahmini karşılaştır ve sonucu göster
        print(f"Soru: {question}")
        print(f"Gerçek Cevap: {true_answer}, Tahmin Edilen: {predicted_answer}\n")

        # Test doğruluğu hesaplamak için (örneğin string eşleşmesiyle)
        if str(predicted_answer) == str(true_answer):
            correct_predictions += 1
        total_predictions += 1

# Genel doğruluğu hesapla
accuracy = 100 * correct_predictions / total_predictions
print(f"Test Doğruluğu: {accuracy:.2f}%")
