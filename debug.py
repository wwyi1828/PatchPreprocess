import pandas as pd
import os

def create_label_df(filename):
    df = pd.read_csv(filename, sep='\t') 
    df = df[['filename']]
    df['type'] = filename.split('.')[0]
    return df

df = pd.concat([create_label_df('TCGA_ACC.txt'), create_label_df('TCGA_PCPG.txt')], ignore_index=True)
df.to_csv('TCGA_Adrenal.csv',index=False)


df = pd.concat([create_label_df('TCGA_LUSC.txt'), create_label_df('TCGA_LUAD.txt')], ignore_index=True)
df.to_csv('TCGA_Lung.csv',index=False)

######Camelyon16 Preprocessing######
df = pd.read_csv('/media/weiyi/My Passport/Data/CAMELYON16/testing/reference.csv', header=None)
filenames = [_+'.tif' for _ in list(df[0])]
types = [_ for _ in list(df[1])]
for _ in os.listdir('/media/weiyi/My Passport/Data/CAMELYON16/training/tumor'):
    filenames.append(_)
    types.append('Tumor')
for _ in os.listdir('/media/weiyi/My Passport/Data/CAMELYON16/training/normal'):
    filenames.append(_)
    types.append('Normal')
df = pd.DataFrame({'filename':filenames, 'type':types})
df.to_csv('CAMELYON16.csv',index=False)







from skimage import color
from scipy.stats import entropy

def calculate_entropy(img):
    if len(img.shape) > 2:
        img = color.rgb2gray(img)  # Convert the image to grayscale if it is RGB
    hist = np.histogram(img.flatten(), bins=256)[0]
    return entropy(hist, base=2)





import h5py

# Open the file with read-only access
# with h5py.File('/pool1/data/TCGA_target/patches/TCGA-WB-A822-01Z-00-DX1.AEF949D0-F3E5-487C-AC7F-9E3D05F06DBC.h5', 'r') as f:
with h5py.File('/media/weiyi/My Passport/Data/Extracted_IMAGES/Camelyon16/patches/tumor_001.h5', 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
    
with h5py.File('/media/weiyi/My Passport/Data/Extracted_IMAGES/Camelyon16/patches/tumor_001.h5', 'r') as f:
    data = list(f['imgs'])
    cord = list(f['coords'])

kk = np.array(data)[np.array(cord)[:,1] == 5248]
kk.shape


for item in data:
    np.mean(item,axis = (0,1))
    calculate_entropy(item)

import openslide
from pathlib import Path
import os
slide = Path('/pool1/data/TCGA_test/TCGA-WB-A822-01Z-00-DX1.AEF949D0-F3E5-487C-AC7F-9E3D05F06DBC.svs')
root_path = Path('/pool1/data/TCGA_Adrenal/')
for slide in os.listdir(root_path):
    slide = openslide.OpenSlide(root_path.joinpath(slide))
    print(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER), slide.level_downsamples)

root_path = Path('/media/weiyi/My Passport/Data/CAMELYON16/training/tumor/')
# root_path = Path('/media/weiyi/My Passport/Data/CAMELYON16/testing/images')
for slide in os.listdir(root_path):
    slide = openslide.OpenSlide(root_path.joinpath(slide))
    print(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER), slide.level_downsamples, end='; ')
    if slide.properties.get('openslide.mpp-x') is not None:
        print(10 / float(slide.properties.get('openslide.mpp-x')))
    else:
        print('None')


root_path = Path('/media/weiyi/My Passport/Data/CAMELYON16/training/normal/')
root_path = Path('/media/weiyi/My Passport/Data/CAMELYON16/testing/images')
for slide in os.listdir(root_path):
    # slide = 'test_011.tif'
    slide = openslide.OpenSlide(root_path.joinpath(slide))
    print(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER), slide.level_downsamples, end='; ')
    if slide.properties.get('openslide.mpp-x') is not None:
        print(10 / float(slide.properties.get('openslide.mpp-x')))
    else:
        print('None')

from scipy.stats import entropy
import numpy as np
import cv2
from skimage import color

def calculate_entropy(img):
    if len(img.shape) > 2:
        img = color.rgb2gray(img)  # convert the image to grayscale if it is RGB
    hist = np.histogram(img.flatten(), bins=256)[0]
    return entropy(hist, base=2)

img = np.array(slide.read_region((5546,165165),level=0,size=(224,224)).convert('RGB'))
calculate_entropy(img[:, :, :3])


level_downsamples = []
dim_0 = slide.level_dimensions[0]

for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
    estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
    level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))



slide.level_downsamples
slide.level_dimensions
magnification = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)

if magnification:
    print(f"Magnification Level: {magnification}x")
else:
    print("Magnification level information is not available for this slide.")


import os
from pathlib import Path
src_folder = Path('/data/data/Extracted_IMAGES/BRACS_test/patches/')

for item in os.listdir(src_folder):
    if len(os.listdir(src_folder.joinpath(item))) < 100:
        print(f'{item}: {len(os.listdir(src_folder.joinpath(item)))}')



import os
from pathlib import Path
import pickle
src_folder = Path('/data/data/Extracted_IMAGES/BRACS_train/patches/')

# Check whether existing slides with few patches
for item in os.listdir(src_folder):
    if len(os.listdir(src_folder.joinpath(item))) < 100:
        print(f'{item}: {len(os.listdir(src_folder.joinpath(item)))}')

total_pngs = list(src_folder.rglob("*.png"))
len(total_pngs)

pkl_file = '/pool1/data/BRACS_R50_train.pkl'
loader = pickle.load(open(pkl_file, 'rb'))
sum([item.pos.size(0) for item in loader])


require_compute = []
total_num_ext_patch = 0
total_num_pkl_patch = 0
for item in loader:
    slide_name = item.slide_index
    num_pkl_patch = item.pos.size(0)
    num_ext_patch = len(list(src_folder.joinpath(slide_name).rglob("*.png")))
    if num_pkl_patch != num_ext_patch:
        print(f'{slide_name} {num_ext_patch} {num_pkl_patch}')
        require_compute.append(slide_name)
    total_num_ext_patch += num_ext_patch
    total_num_pkl_patch += num_pkl_patch



"""
目前来说 timpstamp没有变化。我感觉可以把training的前10个+test前一个+val前一个以及其他还没有出现/处理过的slide放在一起重新处理一遍


"""
from pathlib import Path
import os
import pandas as pd


filename = "svs_files.txt"
svs_files = []
with open(filename, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if parts:
            file_name = parts[-1]
            if file_name.endswith('.svs'):
                svs_files.append(file_name.split('.svs')[0])


csv_slides = list(pd.read_csv('BRACS.csv')['filename'])
split_set = 'train'
raw_path = Path('/data/data/BRACS_WSI')
extracted_path = Path(f'/pool1/data/Extracted_IMAGES/BRACS_{split_set}/patches')

extracted_slides = os.listdir(extracted_path)
raw_slides = list(raw_path.rglob(f"{split_set}*/*.svs"))
raw_slides = list(raw_path.rglob(f"{split_set}*/*"))
raw_slides = [_.stem for _ in raw_slides]

unextracted = set(raw_slides).difference(extracted_slides)
set(extracted_slides).difference(raw_slides)
set(svs_files).difference(raw_slides)
# >>> unextracted
# {'BRACS_1404', 'BRACS_1003717'}




import os
from pathlib import Path
from collections import defaultdict

# 设置基本路径和分割集
base_path = Path(f'/data/data/Extracted_IMAGES/BRACS_{split_set}/patches')

# 使用defaultdict存储每个slide的patches信息
slide_patches = defaultdict(list)

# 遍历base_path下的所有目录和文件
for slide_dir in base_path.iterdir():
    if slide_dir.is_dir():  # 如果是目录，假设它是一个slide目录
        slide_name = slide_dir.name  # 获取slide的名称
        for patch_file in slide_dir.rglob('*.png'):  # 查找所有的patch文件
            # 获取文件的修改时间
            mtime = patch_file.stat().st_mtime
            # 将文件信息添加到slide的列表中
            slide_patches[slide_name].append((mtime, patch_file))

# 对每个slide的patches列表进行排序，确保最新的文件在前
for patches in slide_patches.values():
    patches.sort(key=lambda x: x[0], reverse=True)  # 按mtime降序排序

# 现在，我们有了一个排序后的文件列表，我们可以找到最新的10个patches
latest_patches_info = []

for slide_name, patches in slide_patches.items():
    # 从每个slide的patches中获取前10个（如果有的话）
    top_patches = patches[:10]
    for patch in top_patches:
        latest_patches_info.append((slide_name, patch[1]))  # 存储slide名称和patch文件路径

# 如果您只想要包含最新patches的slide名称，可以这样做：
latest_slides = set(info[0] for info in latest_patches_info)

# 打印结果
for slide in latest_slides:
    print(slide)




from PIL import Image
import os

def check_images(folder_path):
    broken_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            try:
                with Image.open(os.path.join(folder_path, filename)) as img:
                    img.verify()  # Verify the integrity of the file.
            except (IOError, SyntaxError) as e:
                print(f'Broken file: {filename}')
                broken_images.append(filename)
    return broken_images

folder_path = '/pool1/data/Extracted_IMAGES/Camelyon16_test/patches/test_011'  # 替换为您的文件夹路径
broken_images = check_images(folder_path)

print("Broken images:", broken_images)
len(broken_images)
len(os.listdir(folder_path))






# 初始化一个空字典来存储每个类别的计数
category_counts = {}

# 打开并读取文件
with open('tcga.txt', 'r') as file:
    for line in file:
        # 从每行中提取类别名称
        category = line.strip().split('\t')[-1]

        # 更新字典中的计数
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

# 打印出每个类别的计数结果
for category, count in category_counts.items():
    print(f'{category}: {count}')
