import pandas as pd
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--dataset', '-d', help="dataset directory path", type=str, default='DogFaceNet_alignment')

args = parser.parse_args()
main_dir = Path(args.dataset)

df = pd.read_csv(main_dir / 'labels.csv')

for index, row in df.iterrows():
    img_pth = main_dir / 'images' / row['filename']
    
    