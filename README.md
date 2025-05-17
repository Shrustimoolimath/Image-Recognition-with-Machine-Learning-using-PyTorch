Image-Recognition-with-Machine-Learning-using-PyTorch
🧠 Image Recognition with PyTorch 🖼️
A deep learning-based image recognition project using pre-trained ResNet101 architecture from torchvision.models. The model is capable of identifying objects in images using the ImageNet dataset's 1,000 class labels.

📌 Features
🔍 Performs image classification using ResNet101 (101-layer CNN)

🎯 Uses pre-trained weights trained on ImageNet (1.2M images, 1000 categories)

🖼️ Loads and preprocesses local images for inference

📊 Outputs predicted class and confidence score

💡 Easy-to-follow, beginner-friendly PyTorch implementation

🚀 Getting Started
🔧 Requirements
Python 3.x

PyTorch

Torchvision

Pillow (PIL)

Google Colab / Jupyter Notebook

imagenet_classes.txt (downloadable from here)

📦 Installation
bash
Copy
Edit
pip install torch torchvision pillow
🧪 Inference Pipeline
1. 📚 Load Pre-trained Model
python
Copy
Edit
from torchvision import models
resnet = models.resnet101(pretrained=True)
resnet.eval()
2. 🔧 Preprocess the Image
python
Copy
Edit
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open("dog.png")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)
3. 🔍 Run Model on Image
python
Copy
Edit
import torch

out = resnet(batch_t)
_, index = torch.max(out, 1)
4. 📌 Get Prediction Label
python
Copy
Edit
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())
Example Output:

matlab
Copy
Edit
'golden retriever', 96.29%
📁 Project Structure
bash
Copy
Edit
├── image_recognition.ipynb       # Jupyter Notebook with full workflow
├── dog.png                       # Sample image for prediction
├── imagenet_classes.txt          # Class labels for ImageNet dataset
└── README.md                     # Project documentation
🔍 Sample Result
Input Image	Predicted Label	Confidence
	golden retriever	96.29%

🧠 Concepts Used
Transfer Learning

Pretrained CNN (ResNet)

Image Preprocessing & Normalization

Inference Mode in PyTorch

Softmax & Class Probability

🌐 Connect With Me
📍 Made with ❤️ by Your Name
📧 Reach out on LinkedIn or GitHub for collaborations or feedback!

🏷️ Tags
#PyTorch #DeepLearning #ImageRecognition #ComputerVision #AI #MachineLearning #ResNet #TransferLearning #ImageNet
