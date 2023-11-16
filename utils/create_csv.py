import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse

"""
Create csv files for multiclass classification

folder structure:

 Dataset/
    classes.names
    train/
       class1/
           image1.jpg
           image2.jpg
       class2/
           image3.jpg
           image4.jpg
    val/
       class1/
           image1.jpg
           image2.jpg
       class2/
           image3.jpg
           image4.jpg
"""

def process_class(class_id, class_name, dataset_folder, set_):
    class_folder = os.path.join(dataset_folder, class_name)
    if os.path.isdir(class_folder):
        images = os.listdir(class_folder)
        image_paths = []
        labels = []

        for image_name in images:
            image_relative_path = os.path.join(set_, class_name, image_name)
            image_paths.append(image_relative_path)
            labels.append(str(class_id))

        return image_paths, labels
    else:
        return [], []

def main(args):
    root_dir = args.root

    with open(os.path.join(root_dir, 'classes.names'), mode="r") as f:
        class_names = f.read().splitlines()
        classes = dict(enumerate(class_names))
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for set_ in ['train', 'test', 'val', 'train_1k', 'train_1k_stratified']:
            dataset_folder = os.path.join(root_dir, set_)
            results = list(tqdm(executor.map(process_class, classes.keys(), classes.values(),
                                              [dataset_folder] * len(classes), [set_] * len(classes)),
                                total=len(classes)))

            image_paths = [path for paths, _ in results for path in paths]
            labels = [label for _, labels in results for label in labels]
            df = pd.DataFrame({"image_relative_path": image_paths, "label": labels})
            save_dir = os.path.join(root_dir, set_, f"{set_}.csv")
            df.to_csv(save_dir, index=False)
            print(f'CSV saved to {save_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset folders into CSV files.")
    parser.add_argument("--root", type=str, help="Root directory of the dataset")
    parser.add_argument("--workers", type=int, default=16, help="Maximum number of worker")

    args = parser.parse_args()
    main(args)
