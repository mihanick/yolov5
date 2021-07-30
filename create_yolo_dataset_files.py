from pathlib import Path
from sys import path
import numpy as np

from processing import  build_data
from plot_graphics import generate_file
import pandas as pd
from PIL import Image
import os

#################################################################
df, ids = build_data(rebuild=False)
test_fraction = 0.1
test_split_index = round(test_fraction * len(ids))

np.random.seed(42)
np.random.shuffle(ids)

generate_images = True
generate_labels = True

train_images_path = Path("data/dwg/images/train")
train_images_path.mkdir(parents=True,exist_ok=True)
train_labels_path = Path("data/dwg/labels/train")
train_labels_path.mkdir(parents=True,exist_ok=True)
train_desc_file_path = "data/dwg/train.txt"


val_images_path = Path("data/dwg/images/val")
val_images_path.mkdir(parents=True,exist_ok=True)
val_labels_path = Path("data/dwg/labels/val")
val_labels_path.mkdir(parents=True,exist_ok=True)
val_desc_file_path = "data/dwg/val.txt"

with open(train_desc_file_path, "w") as train_desc_file:
        with open(val_desc_file_path, "w") as val_desc_file:
                for i, id in enumerate(ids):
                        desc_file = train_desc_file
                        image_folder = str(train_images_path)
                        label_folder = str(train_labels_path)
                        if i < test_split_index:
                                desc_file = val_desc_file
                                image_folder = str(val_images_path)
                                label_folder = str(val_labels_path)

                        format = 'bmp'
                        image_file_name = "{}/{}.{}".format(image_folder, id, format)
                        label_file_name = "{}/{}.txt".format(label_folder, id)

                        img_size = 512

                        if generate_images:
                                generate_file(
                                        df[df['GroupId'] == id], 
                                        path=image_file_name,
                                        verbose=False, 
                                        draw_dimensions=False, 
                                        draw_texts=False, 
                                        save_file=True,
                                        main_stroke='1',
                                        img_size=img_size,
                                        format=format)

                        desc_file.write("{}\n".format(image_file_name))

                        base_aperture = 6 #px
                        pnt_bb_width = 2*base_aperture + 1
                        pnt_bb_height = 2*base_aperture + 1

                        if generate_labels:
                                dims = df[(df['GroupId'] == id) & (df['ClassName'] == 'AlignedDimension')]

                                labels = []
                                for _, dim_row in dims.iterrows():
                                        # Target will be 5x5 pixels around base point of dimension
                                        # category will be 0 for dim extension line base point
                                        

                                        targets =[
                                                ['0', dim_row['XLine1Point.X'], dim_row['XLine1Point.Y'],pnt_bb_width, pnt_bb_height],
                                                ['0', dim_row['XLine2Point.X'], dim_row['XLine2Point.Y'],pnt_bb_width, pnt_bb_height],
                                        ]

                                        for target in targets:
                                                cat = target[0]
                                                center_x = target[1]
                                                center_y = target[2]

                                                if center_x >= img_size:
                                                        center_x-=base_aperture
                                                if center_x <= 0:
                                                        center_x = base_aperture
                                                if center_y >= img_size:
                                                        center_y-=base_aperture
                                                if center_y <= 0:
                                                        center_y = base_aperture
                                                
                                                duplicated = False
                                                # check for duplicates
                                                match_mask = "{} {:.4f} {:.4f}"
                                                for label in labels:
                                                        if match_mask.format(label[0], label[1], label[2]) == match_mask.format(target[0], target[1], target[2]):
                                                                duplicated = True
                                                                break
                                                if not duplicated:
                                                        labels.append(target)
                                        
                                        dim_x_coords = [dim_row['XLine1Point.X'], dim_row['XLine2Point.X'], dim_row['DimLinePoint.X']] 
                                        dim_y_coords = [dim_row['XLine1Point.Y'], dim_row['XLine2Point.Y'], dim_row['DimLinePoint.Y']] 

                                        x = min(dim_x_coords)
                                        y = min(dim_y_coords)
                                        bb_width = max(dim_x_coords) - x
                                        bb_height = max(dim_y_coords) - y

                                        bb_center_x = x + (bb_width / 2)
                                        bb_center_y = y + (bb_height / 2)

                                        labels.append(['1', bb_center_x, bb_center_y, bb_width, bb_height])
                                                        
                                with open(label_file_name, 'w') as label_file:
                                        for cat, center_x, center_y, bb_width, bb_height in labels:
                                                label_file.write("{} {} {} {} {} \n".format(
                                                        cat,
                                                        center_x / img_size,
                                                        1 - center_y / img_size,
                                                        bb_width / img_size,
                                                        bb_height / img_size
                                                ))

                        # break #debug