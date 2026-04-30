# scripts/analyze/extract_latents.py

import argparse
import pandas as pd
import torch
from tqdm import tqdm

# from monai.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from monai import transforms
from bgp import const

# Import de la fonction qui gère le dataset
from bgp import (get_dataset_from_pd)

class PrintShape(transforms.Transform):
    """
    Un petite classe pour regarder l'évolution de la taille des images
    après chaque transformation.
    """
    def __call__(self, data):
        print("MRI Shape:", data["image"].shape)
        return data

def get_dataloader(csv_path, batch_size=1):
    """
    Fonction de lecture du csv pour sortie loader
    """
    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys=['image_path'], names=['image']),
        transforms.LoadImageD(keys=['image'], image_only=True),
        # PrintShape(),
        transforms.EnsureChannelFirstD(keys=['image']), 
        # PrintShape(),
        transforms.Spacingd(pixdim=const.RESOLUTION, keys=['image']),
        # PrintShape(),
        transforms.ResizeWithPadOrCropd(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        # PrintShape(),
        transforms.ScaleIntensityd(minv=0, maxv=1, keys=['image']),

        # A décommencter si on souhaite sauvegarder et visualiser les images après préprocess
        # transforms.SaveImaged(
        #     keys=["image"],
        #     output_dir="./outputs/ResizePC",
        #     output_postfix="transformed",
        #     output_ext='.nii.gz',
        #     resample=False,
        #     separate_folder=False,
        #     print_log=False # mettre à True pour voir les logs de sauvegarde
        # )
    ])

    df = pd.read_csv(csv_path)
    inferene_set = get_dataset_from_pd(df, transforms_fn)
    
    loader = DataLoader(dataset=inferene_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader


def main():
    parser = argparse.ArgumentParser(description="Extrait les latents des IRM.")
    parser.add_argument("--dataset_csv", required=True, help="Chemin vers le CSV")
    parser.add_argument("--model_path", required=True, help="Chemin vers le modèle *.pth")
    args = parser.parse_args()

    # LECTURE DU DATASET
    loader = get_dataloader(args.dataset_csv)
    
    progress_bar = tqdm(enumerate(loader), total=len(loader))
    for step, batch in progress_bar:
        print("TODO")

if __name__ == "__main__":
    main()