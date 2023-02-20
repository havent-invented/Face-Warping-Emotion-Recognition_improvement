import os
import shutil

data_dir = "./data/SEWA/prep_SEWA/"

for i in os.listdir(data_dir):
    #shutil.rmtree(f"./data/SEWA/prep_SEWA/{i}/cropped_images")
    if not os.path.exists(f"./data/SEWA/prep_SEWA/{i}/cropped_images/"):
        os.makedirs(f"./data/SEWA/prep_SEWA/{i}/cropped_images/")


        os.system(f'ffmpeg -i ./data/SEWA/prep_SEWA/{i}/{i}.avi ./data/SEWA/prep_SEWA/{i}/cropped_images/%06d.jpg')
    