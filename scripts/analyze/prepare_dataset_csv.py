# scripts/analyze/prepare_dataset_csv.py

import os
import pandas as pd
from pathlib import Path
import argparse

def reorder_or_create_columns(df, final_order, fill_value=None):
    """
    Réorganise les colonnes selon final_order. 
    Si certaines colonnes n'existent pas, elles sont créées avec fill_value.    
    df : DataFrame
    final_order : liste des colonnes dans l'ordre souhaité
    fill_value : valeur par défaut pour les nouvelles colonnes (None si pas précisé)
    """
    for col in final_order:
        if col not in df.columns:
            df[col] = fill_value
    return df[final_order]


def prepare_combined_csv(dataset_configs, output_csv, latent_path):
    """
    Traite les 02 datasets (chacun ayant son propre TSV et dossier) 
    et les combine en un seul CSV.
    """
        
    all_dataframes = []

    for config in dataset_configs:
        dataset_name = config['name']
        tsv_path = config['tsv']
        data_dir = config['dir']

        print(f"\nTraitement du dataset : {dataset_name}")
        print(f"Lecture du TSV : {tsv_path}")
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Scanner spécifiquement le dossier de ce dataset
        print(f"Recherche des IRM dans : {data_dir}")
        file_map = {}
        for path in Path(data_dir).rglob("*_space-ANTS80Years3Tbrainbiascorrected_desc-affine-intnorm_T1w.nii.gz"):
            # On extrait le subject_id du nom du fichier (ex: sub-1007_brain.nii.gz -> sub-1007)
            subj_id =  path.name.split("_space-ANTS80Years3Tbrainbiascorrected_desc-affine-intnorm_T1w.nii.gz")[0]
            subj_id = subj_id.split("_run")[0] if "run" in subj_id else subj_id
            file_map[subj_id] = str(path.absolute())

        image_paths = []
        latent_paths = []
        found_flags = []

        # Associer les fichiers aux sujets du TSV
        for subj_id in df['participant_id']:
            if subj_id in file_map:
                img_path = file_map[subj_id]
                image_paths.append(img_path)
                
                # Création du chemin de destination pour le latent
                lat_path = os.path.join(latent_path, f"{subj_id}_latent.npz")
                latent_paths.append(lat_path)
                found_flags.append(True)
            else:
                image_paths.append(None)
                latent_paths.append(None)
                found_flags.append(False)
                print(f"Attention : Aucune IRM trouvée pour {subj_id}")

        # Ajouter les nouvelles colonnes au DataFrame
        df['image_path'] = image_paths
        df['latent_path'] = latent_paths
        df['segm_path'] = image_paths

        # Renommer certaines colones:
        #  pathology_id ==> diagnosis
        df = df.rename(columns={'pathology': 'diagnosis'})
        # participant_id ==> subject_id
        df = df.rename(columns={'participant_id': 'subject_id'})
        
        # On ne garde que les lignes où l'image a bien été trouvée
        df_valid = df[found_flags].copy()
        
        # On ajoute une colonne pour garder une trace de l'origine
        df_valid['source_dataset'] = dataset_name 
        df_valid['image_uid'] = df_valid['subject_id']
        
        # Normalisation du sexe
        df_valid['sex_norm'] = df['sex'].map({'F': 1, 'M': 0})

        # Normalisation min-max entre 12 et 84
        df_valid['age'] = (df_valid['age']*12).round(1) if dataset_name in "mtbi-koala" else df_valid['age'] # passer les ages de koala en mois pour correspondre à daufin
        age_min = df_valid['age'].min()
        age_max = df_valid['age'].max()
        age = df_valid['age']

        df_valid['age_norm'] = ((age - age_min) / (age_max - age_min) * (84 - 12) + 12).round(1)

        all_dataframes.append(df_valid)
        print(f"  -> {len(df_valid)} sujets valides trouvés pour {dataset_name}.")

    # FUSION DES 02 DATASETS
    if all_dataframes:
        print("\nFusion des données")
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Rajouter les champs manquants
        if 'split' not in final_df.columns:
            final_df['split'] = 'val' #    
        
        # mettre une valeur par defaut à 0 pour time_most_injury_days
        final_df['time_post_injury_days'] = final_df['time_post_injury_days'].fillna(0)

        # Ordre des colones que j'ai trouvé dans le README (à peu près)
        final_order = [
            'subject_id','image_uid','split','sex', 'sex_norm', 'age', 'age_norm', 
            'diagnosis','last_diagnosis','time_post_injury_days','image_path','segm_path','latent_path',            
            'source_dataset','recruitment_site',	'mri_site', 'institution',
            'DWI_acquired',	'scan_date',	'injury_time',
            'GCS',	'T1_acquired',	'T2_acquired',	'QSM_acquired',	'T2star_phase_acquired',
            'radiology_result',	'non_significant_clinical_findings',
            'head_size','accumbens_area','amygdala','brain_stem','caudate','cerebellum_cortex','cerebellum_white_matter',
            'cerebral_cortex','cerebral_white_matter','csf','fourth_ventricle','hippocampus','inferior_lateral_ventricle',
            'lateral_ventricle','pallidum','putamen','thalamus','third_ventricle','ventral_dc'
        ]

        # Réorganiser et créer les colonnes manquantes avec 0
        final_df_output = reorder_or_create_columns(final_df, final_order, fill_value=0.0)
            
        # Sauvegarde finale
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        final_df_output.to_csv(output_csv, index=False)
        print(f"Terminé ! CSV global créé avec succès : {output_csv} (Total : {len(final_df_output)} sujets)")
    else:
        print("Erreur : Aucune donnée valide trouvée dans l'ensemble des datasets.")

def main():
    parser = argparse.ArgumentParser(description="Création du dataset en csv.")
    parser.add_argument("--root_path", required=True, help="Chemin vers le répertoire parent des 02 datasets")
    parser.add_argument("--output_csv", required=True, help="Chemin de sauvegarde du dataset csv")
    parser.add_argument("--latent_path", required=True, help="Chemin de sauvegarde des latents après extraction")
    args = parser.parse_args()
    
    configs = [
        {
            "name": "hc-daufin", 
            "tsv": f"{args.root_path}/hc-daufin/participants.tsv", 
            "dir": f"{args.root_path}/hc-daufin/derivatives/brainprep"
        },
        {
            "name": "mtbi-koala", 
            "tsv": f"{args.root_path}/mtbi-koala/participants.tsv", 
            "dir": f"{args.root_path}/mtbi-koala/derivatives/brainprep"
        }
    ]
    
    prepare_combined_csv(configs, args.output_csv, args.latent_path)

if __name__ == "__main__":
    main()