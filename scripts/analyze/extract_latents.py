# scripts/analyze/extract_latents.py

import argparse
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from monai import transforms
from bgp import const
from bgp import (init_autoencoder, get_dataset_from_pd)

class PrintShape(transforms.Transform):
    """
    Permet de voir l'évolution de la taille des images
    après chaque transformation.
    """
    def __call__(self, data):
        print("MRI Shape:", data["image"].shape)
        return data

def get_dataloader(csv_path, batch_size=1):
    """
    Fonction de lecture du csv pour sortie loader avec transformations Monai
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
        #     output_dir="outputs/ResizedMRI",
        #     output_postfix="transformed",
        #     output_ext='.nii.gz',
        #     resample=False,
        #     separate_folder=False,
        #     print_log=False # mettre à True pour voir les logs de sauvegarde
        # )
    ])

    df = pd.read_csv(csv_path)
    inference_set = get_dataset_from_pd(df, transforms_fn)
    
    loader = DataLoader(dataset=inference_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader


def main():
    parser = argparse.ArgumentParser(description="Extrait les latents des IRM.")
    parser.add_argument("--dataset_csv", required=True, help="Chemin vers le CSV")
    parser.add_argument("--model_path", required=True, help="Chemin vers le modèle *.pth")
    args = parser.parse_args()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {DEVICE}")
    
    # Chargement du modèle
    print("Initialisation de l'AutoencoderKL ...")
    
    # Chargement des poids depuis le fichier *.pth fourni
    print(f"Chargement des poids depuis {args.model_path}...")
    autoencoder = init_autoencoder(args.model_path).to(DEVICE)
    autoencoder.eval()

    # Lecture du dataset
    loader = get_dataloader(args.dataset_csv)
    
    # Extraction des latents
    
    # # A décommenter si utile de visualiser les latents en .nii.gz
    # save_transform = transforms.SaveImage(
    #     output_dir="outputs/latentViz",
    #     output_postfix="latent",
    #     output_ext=".nii.gz",
    #     resample=False,
    #     separate_folder=False,
    #     print_log=False
    # )    
    
    progress_bar = tqdm(enumerate(loader), total=len(loader))
    with torch.no_grad():
        for step, batch in progress_bar:        
            inputs = batch["image"].to(DEVICE)
            output_path = batch["latent_path"][0]

            # Création du dossier de destination pour les latents
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            z_mu, _ = autoencoder.encode(inputs)
            latent_numpy = z_mu.cpu().numpy()
            
            # Sauvegarder les latents en *.npz
            np.savez_compressed(output_path, latent=latent_numpy)
            print(f"[{step+1}/{len(loader)}] Latent sauvegardé : {output_path}")
            
            # # Décommenter pour sauvegarder les latent en *.nii.gz
            # latent_tensor = transforms.ToTensor()(np.squeeze(latent_numpy))
            # save_transform(latent_tensor)

if __name__ == "__main__":
    main()