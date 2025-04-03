from PIL import Image
import os

def resize_all(input_folder, output_folder, size=(96, 96)):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            path = os.path.join(input_folder, filename)
            img = Image.open(path).convert("L")
            img = img.resize(size)
            img.save(os.path.join(output_folder, filename))

# Base path to downloaded images
base_input = "data_temp"
base_output = "data_set"

# Dataset split folders: train, val, test
splits = ["train", "val", "test"]

for split in splits:
    in_dir = os.path.join(base_input, split, "screwdriver")
    out_dir = os.path.join(base_output, split, "screwdriver")
    #in_dir = os.path.join(base_input, split, "not-screwdriver")
    #out_dir = os.path.join(base_output, split, "not-screwdriver")
    resize_all(in_dir, out_dir)
