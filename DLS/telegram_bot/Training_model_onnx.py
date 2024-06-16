import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import copy
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
""" Transformer Net """
class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),  # Добавлено
            ConvBlock(128, 256, kernel_size=3, stride=2),
            ConvBlock(256, 384, kernel_size=3, stride=2),  # Изменено
            ResidualBlock(384),
            ResidualBlock(384),
            ResidualBlock(384),
            ResidualBlock(384),
            ResidualBlock(384),
            ConvBlock(384,256, kernel_size=3, upsample=True),  # Изменено
            ConvBlock(256, 128, kernel_size=3, upsample=True),  # Добавлено
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False),
        )

    def forward(self, x):
        return self.model(x)
""" Components of Transformer Net """
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False),
        )

    def forward(self, x):
        return self.block(x) + x
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2), nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        return x
def gram_matrix(tensor):
    b, ch, h, w = tensor.size()
    #features = tensor.view(b, ch, h * w)
    temp_features = tensor.clone().view(b, ch, h * w)
    gram = torch.matmul(temp_features, temp_features.transpose(1, 2))
    #gram = torch.matmul(features, features.transpose(1, 2))
    gram = gram / (ch * h * w)
    return gram
class ContentLoss(nn.Module):
    def __init__(self, target_feature):
        super(ContentLoss, self).__init__()
        self.target = target_feature.detach()

    def forward(self, input):
        return F.mse_loss(input, self.target)
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss_fn = nn.MSELoss()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = self.loss_fn(G, self.target)
        return self.loss
# Гиперпараметры
style_image_path = "/kaggle/input/impressionists-monet-vangogh/Sn_VanGogh.jpg"
content_image_path = "/kaggle/input/styletransfer/input.jpg"
output_image_path = "output_image.jpg"
learning_rate = 1e-5
num_epochs =1500
batch_size = 1

# Загрузка изображений стиля и содержимого
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

style_image = Image.open(style_image_path)
style_image = transform(style_image).unsqueeze(0).to(device)
style_image = style_image.repeat(batch_size, 1, 1, 1)

content_image = Image.open(content_image_path)
content_image = transform(content_image).unsqueeze(0).to(device)

style_weights = [0.05,0.05,0.2,0.3,0.4,0.5] #[1e3]#, 1e3, 1e3, 1e3, 1e3]  # Вес для каждого слоя стиля
content_weight = [1]#,1e5,1e5,1e5]  # Вес для слоя содержимого
# Определение модели и оптимизатора
transformer = TransformerNet().to(device)
optimizer = optim.AdamW(transformer.parameters(), lr=learning_rate)

# Загрузка предобученной модели VGG
#vgg = models.vgg19(pretrained=True).features.to(device).eval()
vgg=models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False


# Извлечение признаков стиля из изображения стиля
style_layers =['20','22','26','29','33','35']#['3', '8', '17', '26','33'['1','6','11','20','29']#['19','21','26','29','33','35']#['0', '5', '10', '19','21', '28','33','35'] #'26','29', ['31','33','35'] 
style_features = []
x = style_image
for name, layer in vgg._modules.items():
    x = layer(x)
    if name in style_layers:
        style_features.append(x)
style_losses = [StyleLoss(style_feature) for style_feature in style_features]
# Извлечение признаков содержимого из изображения содержимого
content_layers =['24'] #'3', '8', '17'
content_features = []
x_content = content_image
for name, layer in vgg._modules.items():
    x_content = layer(x_content)
    if name in content_layers:
        content_features.append(x_content)
content_losses = [ContentLoss(content_feature) for content_feature in content_features]
#Определение модели
# Обучение модели на одном изображении
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = transformer(content_image)
    # Вычисление потерь стиля
    style_loss = torch.tensor(0.0, device=device, requires_grad=True)
    x = output
    #for i, layer in enumerate(vgg._modules.values()):
    for name, layer in vgg._modules.items():
        x = layer(x)
    #if str(i) in style_layers:
        if name in style_layers:
                #target_feature = style_losses[style_layers.index(str(i))].target
                #if x.size()[1:] != target_feature.size()[1:]:
                    #target_feature = torch.nn.functional.interpolate(target_feature, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
                #    x = torch.nn.functional.interpolate(x, size=(target_feature.size(2), target_feature.size(3)), mode='bilinear', align_corners=False)
                idx = style_layers.index(name)
                style_loss = style_loss + style_losses[idx](x) * style_weights[idx]
                #style_loss = style_loss + style_losses[style_layers.index(name)](x)

    # Вычисление потерь содержания
    content_loss = torch.tensor(0.0, device=device)
    x_content = output
    for name, layer in vgg._modules.items():
        x_content = layer(x_content)
        if name in content_layers:
            #target_feature = content_losses[content_layers.index(name)].target
            #if x.size()[2:] != target_feature.size()[2:]:
            #    x = torch.nn.functional.interpolate(x, size=target_feature.size()[2:], mode='bilinear', align_corners=False)
            idx = content_layers.index(name)
            content_loss = content_loss + content_losses[idx](x_content) * content_weight[idx]
            #content_loss =content_loss + content_losses[content_layers.index(name)](x_content)
                        
    total_loss = 15*style_loss + content_loss
    #total_loss = style_loss
    total_loss.backward()
    optimizer.step()

    if epoch%250==0:
        print( time.strftime("%H:%M", time.localtime()),f"Epoch {epoch+1}/{num_epochs}, Style Loss: {style_loss.item()}, Content Loss: {content_loss.item()}")
    #print(f"Epoch {epoch+1}/{num_epochs}, Content Loss: {style_loss.item()}")


# Сохранение обученной модели
torch.save(transformer.state_dict(), "style_transfer_model.pth")

# Применение модели к содержательному изображению
# Функция для денормализации изображения и преобразования в PIL Image
def denormalize(tensor):
    tensor = tensor.mul(0.5).add(0.5).clamp(0, 1)  # Undo normalization
    tensor = tensor.squeeze(0)  # Remove the batch dimension
    tensor = tensor.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
    return Image.fromarray((tensor * 255).astype('uint8'))  # Convert to PIL Image

#content_image_path='/kaggle/input/impressionists-monet-vangogh/input.jpg' #'/kaggle/input/landscape-pictures/00000000_(2).jpg' #/kaggle/input/landscape-pictures/00000000_(2).jpg /kaggle/input/landscape-pictures/00000001_(5).jpg /kaggle/input/landscape-pictures/00000000_(2).jpg
content_image_path='/kaggle/input/impressionists-monet-vangogh/input.jpg' #'/kaggle/input/landscape-pictures/00000000_(2).jpg'
output_image_path='output.jpg'
content_image = Image.open(content_image_path)
w,h=content_image.size
content_image = transform(content_image).unsqueeze(0).to(device)
with torch.no_grad():
    output = transformer(content_image).cpu().squeeze(0)
    output_image = denormalize(output)
    output_image=output_image.resize((w,h))
    #output = output * 0.5 + 0.5 # Денормализация
    #output_image = transforms.ToPILImage()(output)

# Сохранение результата
output_image.save(output_image_path)

# Отображение результата
plt.imshow(output_image)
plt.axis('off')
plt.show()


!pip install onnx
!pip install onnxscript
!pip install onnxruntime

# Экспорт модели в формат ONNX
dummy_input = torch.randn(1, 3, h, w, device='cuda' if torch.cuda.is_available() else 'cpu')
torch.onnx.export(transformer, dummy_input, "style_transfer.onnx", verbose=True)
import onnx
import onnxruntime
import numpy as np

def load_image(img_path, size):
    image = Image.open(img_path).convert('RGB')
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    image = in_transform(image).unsqueeze(0)
    return image.requires_grad_(False)


# ЗагрузкаONNX
onnx_model_path = "/kaggle/working/style_transfer.onnx"
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)


ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Получить ожидаемые размеры ONNX
input_shape = ort_session.get_inputs()[0].shape
expected_height = input_shape[2]
expected_width = input_shape[3]
print(f"Expected input shape: {input_shape}")

# Загрузка изображений с изменением размера
content_img_path ="/kaggle/input/styletransfer/input.jpg"# "/kaggle/input/styletransfer/input.jpg"  #/kaggle/input/landscape-pictures/00000001_(5).jpg
style_img_path = "/kaggle/input/impressionists-monet-vangogh/training/VanGogh/207189.jpg"
to_sizes=Image.open(content_img_path)
w,h =to_sizes.size
content_img = load_image(content_img_path, (expected_height, expected_width)).to('cuda' if torch.cuda.is_available() else 'cpu')
style_img = load_image(style_img_path, (expected_height, expected_width)).to('cuda' if torch.cuda.is_available() else 'cpu')

def run_onnx_model(ort_session, input_tensor):
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    return torch.tensor(ort_outs[0])

# Run
output = run_onnx_model(ort_session, content_img)
final_image = denormalize(output)

print(output.shape)
#final_image = transforms.ToPILImage()(final_image)#.squeeze(0)) # tensor_to_image(output)
final_image=final_image.resize((w,h))
display(final_image)
final_image.save("output_image_onnx.jpg")
