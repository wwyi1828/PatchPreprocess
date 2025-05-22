import pandas as pd
from pathlib import Path
import os
import openslide

BRACS_path = Path('/data/data/BRACS_WSI')
svs_files = list(BRACS_path.rglob('*.svs')) 


filenames = []
types= []
subtypes = []
error_files = []
for svs_file in svs_files:
    filenames.append(str(svs_file).split('/')[-1])
    subtypes.append(str(svs_file).split('/')[-2])
    types.append(str(svs_file).split('/')[-3])

    try:
        slide = openslide.OpenSlide(str(svs_file))
    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
        print(f"Error processing file: {svs_file}")
        error_files.append(svs_file)
    print(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER), slide.level_downsamples, end='; \n')
    if slide.properties.get('openslide.mpp-x') is not None:
        print(10 / float(slide.properties.get('openslide.mpp-x')))
    else:
        print('None')

BRACS_type = pd.DataFrame({'filename':filenames, 'type':types})
BRACS_subtype = pd.DataFrame({'filename':filenames, 'type':subtypes})

# BRACS_type.to_csv('BRACS.csv',index=False)
# BRACS_subtype.to_csv('BRACS_subtype.csv',index=False)

# BRACS_path = Path('/data/data/BRACS_WSI')
BRACS_path = Path('/data/data/TCGA_LUNG_None')
svs_files = list(BRACS_path.rglob('*.svs')) 
for svs_file in svs_files:
    slide = openslide.OpenSlide(str(svs_file))
    slide.properties.get('openslide.vendor') 


# TCGA BRAC
tsv_file_path = '/data/data/TCGA_clinical/brca_tcga_pan_can_atlas_2018_clinical_data.tsv'
svs_file_path = '/data/Download/TCGA_BRAC_unique'
tsv_data = pd.read_csv(tsv_file_path, sep='\t')
print(tsv_data['Tumor Type'].value_counts(normalize=False))
major_types = ['Infiltrating Ductal Carcinoma', 'Infiltrating Lobular Carcinoma']
tsv_data['Tumor Type'] = tsv_data['Tumor Type'].apply(lambda x: x if x in major_types else 'Other')
print(tsv_data['Tumor Type'].value_counts())
tsv_data[['Sample ID', 'Tumor Type']]
sample_to_label = dict(zip(tsv_data['Sample ID'], tsv_data['Tumor Type']))

svs_label_dict = {}
for svs_file in os.listdir(svs_file_path):
    sample_id = svs_file.split('.')[0][:-8]
    label = sample_to_label.get(sample_id, 'Unknown')  # 如果找不到对应的 Sample ID，标记为 'Unknown'
    svs_label_dict[svs_file] = label
svs_label_df = pd.DataFrame(list(svs_label_dict.items()), columns=['filename', 'type'])
# svs_label_df = svs_label_df[svs_label_df['type'] != 'Other']
svs_label_df.to_csv('TCGA_BRAC.csv', index=False)


# TCGA COAD
tsv_file_path = '/data/data/TCGA_clinical/coadread_tcga_pan_can_atlas_2018_clinical_data.tsv'
svs_file_path = '/data/Download/TCGA_COAD'
tsv_data = pd.read_csv(tsv_file_path, sep='\t')
print(tsv_data['Tumor Type'].value_counts(normalize=False))
major_types = list(tsv_data['Tumor Type'].value_counts(normalize=False).keys())
tsv_data['Tumor Type'] = tsv_data['Tumor Type'].apply(lambda x: x if x in major_types else 'Other')
print(tsv_data['Tumor Type'].value_counts())
tsv_data[['Sample ID', 'Tumor Type']]
sample_to_label = dict(zip(tsv_data['Sample ID'], tsv_data['Tumor Type']))

svs_label_dict = {}
for svs_file in os.listdir(svs_file_path):
    sample_id = svs_file.split('.')[0][:-8]
    label = sample_to_label.get(sample_id, 'Unknown')  # 如果找不到对应的 Sample ID，标记为 'Unknown'
    svs_label_dict[svs_file] = label
svs_label_df = pd.DataFrame(list(svs_label_dict.items()), columns=['filename', 'type'])
# svs_label_df = svs_label_df[svs_label_df['type'] != 'Other']
print(svs_label_df['type'].value_counts())
svs_label_df.to_csv('TCGA_COAD.csv', index=False)