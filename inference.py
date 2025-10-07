import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img = Image.open("sample.png")
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)
    predicted = torch.argmax(output, 1)
    print(f"预测类别：{predicted.item()}")
