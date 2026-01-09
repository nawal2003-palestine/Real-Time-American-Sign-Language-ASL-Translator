import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
EXTERNAL_DATASET_PATH = "../asl_alphabet_train/asl_alphabet_train"  # Chemin vers le dossier principal
OUTPUT_FILE = "./data.pickle"

# Initialiser MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

data = []
labels = []

print("🔍 Recherche du dataset...")

# Vérifier si le dossier existe
if not os.path.exists(EXTERNAL_DATASET_PATH):
    print(f"❌ Erreur : Le dossier '{EXTERNAL_DATASET_PATH}' n'existe pas")
    print("\nVeuillez vérifier le chemin du dataset.")
    exit()

# Vérifier si c'est bien un dossier avec des sous-dossiers
subdirs = [d for d in os.listdir(EXTERNAL_DATASET_PATH) 
           if os.path.isdir(os.path.join(EXTERNAL_DATASET_PATH, d))]

if len(subdirs) == 0:
    print(f"❌ Erreur : Aucun sous-dossier trouvé dans '{EXTERNAL_DATASET_PATH}'")
    print("Le dossier doit contenir des sous-dossiers pour chaque lettre (A, B, C, etc.)")
    exit()

# Filtrer uniquement les dossiers de lettres (ignorer other, nothing, space, etc.)
classes = sorted([d for d in subdirs if len(d) == 1 and d.isalpha()])

if len(classes) == 0:
    print("❌ Aucune classe de lettre trouvée (A-Z)")
    print(f"Dossiers trouvés : {subdirs}")
    exit()

print(f"✅ Trouvé {len(classes)} lettres : {classes}")
print("\n🚀 Traitement des images...")

# Parcourir chaque classe
for class_name in classes:
    class_dir = os.path.join(EXTERNAL_DATASET_PATH, class_name)
    image_files = [f for f in os.listdir(class_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n📁 Lettre '{class_name}' : {len(image_files)} images")
    
    # Limiter à 100 images par classe pour un traitement plus rapide
    image_files = image_files[:100]
    
    successful = 0
    
    # Traiter chaque image avec barre de progression
    for img_file in tqdm(image_files, desc=f"Traitement {class_name}"):
        img_path = os.path.join(class_dir, img_file)
        
        try:
            # Lire l'image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Convertir en RGB (MediaPipe utilise RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Détecter les mains
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                # Extraire les coordonnées des landmarks
                data_aux = []
                x_coords = []
                y_coords = []
                
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x_coords.append(landmark.x)
                        y_coords.append(landmark.y)
                    
                    # Normaliser par rapport au point minimal
                    min_x = min(x_coords)
                    min_y = min(y_coords)
                    
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min_x)
                        data_aux.append(landmark.y - min_y)
                
                # Ajouter aux données
                data.append(data_aux)
                labels.append(class_name)
                successful += 1
                
        except Exception as e:
            continue
    
    print(f"✅ {successful}/{len(image_files)} images traitées avec succès pour '{class_name}'")

hands.close()

# Sauvegarder les données
print(f"\n💾 Sauvegarde de {len(data)} échantillons...")

if len(data) == 0:
    print("❌ Aucune donnée à sauvegarder ! Vérifiez que les images contiennent bien des mains.")
else:
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    
    print(f"✅ Dataset créé avec succès !")
    print(f"📊 Total : {len(data)} échantillons")
    print(f"📝 Lettres : {sorted(set(labels))}")
    print(f"💾 Fichier : {OUTPUT_FILE}")