import torch
from torchvision import models, transforms
from PIL import Image
import requests
model = models.resnet18(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 3. Load local image
img = Image.open("image.jpg").convert('RGB')
img_tensor = preprocess(img)
batch = img_tensor.unsqueeze(0)

# 4. Make prediction
with torch.no_grad():
    output = model(batch)

# 5. Get predicted class
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top10_prob, top10_index = torch.topk(probabilities, 10)

# 6. Load class labels
labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(labels_url).json()

# 7. Print results
print("Top 10 predictions:")
for i in range(10):
    print(f"{labels[top10_index[i]]}: {top10_prob[i].item()*100:.2f}%")