model.eval()  # 设置为评估模式

# 创建一个纯白的图像 (C, H, W)
white_image = torch.ones(1, 3, 224, 224) * 255  # (batch_size, channels, height, width)

# 定义预处理转换
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 应用预处理
white_image /= 255.0  # 缩放到[0, 1]
white_image = transform(white_image)  # 标准化

# 获取模型输出
with torch.no_grad():  # 关闭梯度计算
    output = model(white_image)
    predicted = torch.argmax(output, dim=1)

print("Output:", output)
print("Predicted class index:", predicted.item())
torch.save(output, 'white_R50feats.pt')