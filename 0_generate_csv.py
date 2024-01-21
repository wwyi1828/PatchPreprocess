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