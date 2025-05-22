import pandas as pd
import os
import shutil

# 读取csv数据并找出Other类型的文件
data = pd.read_csv('labels/TCGA_BRAC.csv')
type_counts = data['type'].value_counts()
print(type_counts)

# 找出Other类型的基础文件名（不包含UUID部分）
other_files = data[data['type'] == 'Other']['filename'].tolist()
other_base_names = [f.split('.')[0] for f in other_files]  # 只取第一个点之前的部分

# 源目录和目标目录
src_dir = "/pool2/data/TCGA_BRAC_UNI"
dst_dir = "/pool2/data/TCGA_BRAC_UNI_other" 

# 创建目标目录（如果不存在）
os.makedirs(dst_dir, exist_ok=True)

# 获取源目录中的所有文件
existing_files = os.listdir(src_dir)

# 找到匹配的文件
files_to_move = []
for base_name in other_base_names:
    matching_files = [f for f in existing_files if f.startswith(base_name)]
    files_to_move.extend(matching_files)

# 打印将要移动的文件列表
print("将要移动以下文件：")
for file in files_to_move:
    print(file)

# 确认是否继续
confirm = input("\n确认要移动这些文件吗? (y/n): ")

if confirm.lower() == 'y':
    # 移动文件
    moved_count = 0
    not_found_count = 0
    
    for file in files_to_move:
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(dst_dir, file)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f"已移动: {file}")
            moved_count += 1
        else:
            print(f"未找到文件: {file}")
            not_found_count += 1
    
    print(f"\n移动完成!")
    print(f"成功移动: {moved_count} 个文件")
    print(f"未找到: {not_found_count} 个文件")
else:
    print("操作已取消")