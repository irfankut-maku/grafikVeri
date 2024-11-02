import os
import json
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # JSON dosyasını yükleyin
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item[
            'image']  # Burada img_path yalnızca dosya adını içermelidir, örn: 'table_chart_yearly_revenue.png'

        # Dosya yolunu oluşturun
        image_path = os.path.join(self.img_dir, img_path)

        # Görseli açın ve RGB formatına dönüştürün
        image = Image.open(image_path).convert('RGB')

        # Görsel üzerinde dönüşüm uygulayın (varsa)
        if self.transform:
            image = self.transform(image)

        # Soru ve cevap çiftlerini alın
        qa_pairs = item['qa_pairs']

        # Eğitim için birden fazla soru-cevap varsa ilkini kullanın
        question = qa_pairs[0]['question']
        answer = qa_pairs[0]['answer']

        return image, question, answer
