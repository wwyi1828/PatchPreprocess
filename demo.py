from transformers import AutoImageProcessor
from torch.utils.data import DataLoader
from transformers import ViTMAEModel, AutoModel

# 初始化image_processor
model_config = 'facebook/vit-mae-large'
model_config = 'facebook/dinov2-large'
image_processor = AutoImageProcessor.from_pretrained(model_config)
# model = ViTMAEModel.from_pretrained(model_config)
model = AutoModel.from_pretrained(model_config)
model.eval()  # 设置为评估模式

dataset = PNGDataset(f'{src_folder}{slide_h5}', image_processor)

# 创建DataLoader
data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

features_list = []
device = torch.device("cuda")
model = model.cuda()
with torch.no_grad():
    for batch in data_loader:
        images = batch[0]
        images = images.to(device)
        outputs = model(images)
        last_hidden_states = outputs.last_hidden_state
        img_representations = last_hidden_states[:, 0, :] # batch_size*num_hidden
        # img_representations = torch.mean(last_hidden_states, dim=1)

        features_list.append(img_representations)


# Use a pipeline as a high-level helper
from PIL import Image
from transformers import pipeline
import torch

pipe = pipeline("zero-shot-image-classification", model="vinid/plip")# Load model directly
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

processor = AutoProcessor.from_pretrained("vinid/plip")
model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")

image_path = "/pool1/data/Extracted_IMAGES/Camelyon16/patches/normal_001/32928_62944.png"  # 替换为你的图像URL
image = Image.open(image_path)

# 处理图像
inputs = processor(images=image, return_tensors="pt")

# 获取图像特征
with torch.no_grad():
    features = model.vision_model(**inputs).pooler_output  # 使用 pooler_output 获取编码后的图像特征

# features 现在包含了图像的特征向量
print(features)

