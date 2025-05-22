import os
import zipfile

# patient_035_node_2.tif   bad CRC 6167bfc4  (should be bab82de7) #deleted
# patient_194_node_3.tif   bad CRC 5d997aed  (should be 13373d35)
# patient_105_node_2.tif   bad CRC 1ef7ed0b  (should be 53f94d25)
#   inflating: patient_132_node_1.tif  
#   error:  invalid compressed data to inflate

def unzip_and_delete(directory):
    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                # 完整的文件路径
                file_path = os.path.join(root, file)
                
                # 解压 ZIP 文件
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # 解压到当前目录
                    zip_ref.extractall(root)
                    print(f'Unzipped {file_path}')
                
                os.remove(file_path)
                print(f'Deleted {file_path}')

if __name__ == '__main__':
    directory = '/data/data/CAMELYON17/testing/patients'
    unzip_and_delete(directory)
    # for i in [1, 2, 3, 4]:
    #     center_directory = os.path.join(base_directory, f'center_{i}')
    #     print(f'Processing directory: {center_directory}')
    #     unzip_and_delete(center_directory)
        # directory = center_directory



import pandas as pd
df = pd.read_csv('CAMELYON17.csv')

# 过滤掉包含 '.zip' 的行
filtered_df = df[~df['patient'].str.contains('\.zip')]


filtered_df = filtered_df.rename(columns={'patient': 'file', 'stage': 'type'})
# 将结果保存到新的 CSV 文件
filtered_df.to_csv('CAMELYON17.csv', index=False)