import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame

pygame.init()
alert_sound = pygame.mixer.Sound("./assets/Sound.mp3")


# Fonction pour calculer le ratio de l'œil (Eye Aspect Ratio - EAR)
def eye_aspect_ratio(eye):
    # Calcul de la distance entre les points verticaux des yeux
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Calcul de la distance horizontale
    C = dist.euclidean(eye[0], eye[3])
    # Calcul du EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Définition des constantes
EAR_THRESHOLD = 0.25  # Seuil pour la fermeture de l'œil
EAR_FRAMES = 20  # Nombre de frames consécutives indiquant la fatigue

# Initialisation du compteur
counter = 0  # Compteur pour le nombre de frames avec les yeux fermés

# Charger le détecteur de visage et le prédicteur de points de repère
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./module/shape_predictor_68_face_landmarks.dat")

# Indices pour les yeux dans le modèle de landmarks de Dlib
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Démarrer la capture vidéo
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Lire une image depuis la caméra
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Détecter les visages dans l'image
    faces = detector(gray)
    
    for face in faces:
        # Obtenir les points de repère faciaux
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Extraire les coordonnées des deux yeux
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Calculer le EAR pour chaque œil
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # Prendre la moyenne du EAR des deux yeux
        ear = (leftEAR + rightEAR) / 2.0
        
        # Dessiner les contours des yeux
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Vérifier si l'EAR est en dessous du seuil
        if ear < EAR_THRESHOLD:
            counter += 1
            # Si les yeux sont fermés depuis suffisamment longtemps, afficher "Fatigue détectée"
            if counter >= EAR_FRAMES:
                alert_sound.play()
                cv2.putText(frame, "FATIGUE DETECTEE!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            counter = 0
    
    # Afficher l'image avec les annotations
    cv2.imshow("Detection de Fatigue", frame)
    
    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()

