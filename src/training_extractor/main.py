import librosa
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import os

def extract_features(audio_data, sample_rate, n_mfcc=40):
    """
    Extrae un conjunto de características de una señal de audio.
    Usamos la media de los MFCCs, Chroma y Mel Spectrogram.
    """
    features = {}
    
    # Asegurarse de que el audio sea flotante (requerido por librosa)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # MFCCs (Coeficientes Cepstrales en la Frecuencia Mel)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    for i, mfcc_val in enumerate(mfccs_mean):
        features[f'mfcc_{i+1}_mean'] = mfcc_val

    # Chroma (Características de Tono)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    features['chroma_mean'] = np.mean(chroma)

    # Mel Spectrogram (Espectrograma en escala Mel)
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    features['mel_mean'] = np.mean(mel)
    
    # Contraste Espectral (Spectral Contrast)
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    features['contrast_mean'] = np.mean(contrast)

    return features

# --- Script Principal ---

print("Iniciando el proceso...")

os.environ['TRANSFORMERS_TRUST_REMOTE_CODE'] = '1'
os.environ['HF_DATASETS_DISABLE_TORCHCODEC'] = '1'

dataset_name = "stapesai/ssi-speech-emotion-recognition"
SAMPLING_RATE = 16000

# 1. CARGAR DATOS - Sin Audio wrapper para evitar decodificación
print("Cargando el dataset...")
try:
    dataset = load_dataset(dataset_name, 'default', split='train')
except Exception as e:
    print(f"Error: {e}")
    dataset = load_dataset(dataset_name, split='train')

print(f"Dataset cargado. Total de muestras: {len(dataset)}")

# 2. EXTRAER CARACTERÍSTICAS
processed_data = []
print("Extrayendo características...")

# Acceder directamente a la tabla PyArrow para evitar decodificación
arrow_table = dataset.data

# Inspeccionar la estructura
print(f"Columnas disponibles: {arrow_table.column_names}")

# Procesar todas las muestras
for idx in tqdm(range(arrow_table.num_rows)):
    try:
        # Acceder directamente a la tabla PyArrow sin formateo
        row_dict = {
            name: arrow_table.column(name)[idx].as_py() 
            for name in arrow_table.column_names
        }
        
        # Obtener info del audio y emoción
        file_path_info = row_dict.get('file_path')  # Dict con 'bytes' y 'path'
        emotion_label = row_dict.get('emotion')
        
        # Extraer audio bytes
        if isinstance(file_path_info, dict) and 'bytes' in file_path_info:
            audio_bytes = file_path_info['bytes']
            
            # Guardar temporalmente y cargar con librosa
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                audio_data, sample_rate = librosa.load(tmp_path, sr=SAMPLING_RATE)
                
                # Extraer características
                features = extract_features(audio_data, sample_rate)
                features['emotion'] = emotion_label
                processed_data.append(features)
            finally:
                os.remove(tmp_path)
        
    except Exception as e:
        pass  # Ignorar errores

print(f"\nSe procesaron {len(processed_data)} muestras")

# 3. CREAR Y GUARDAR EL CSV y XLSX
if processed_data:
    print("Creando DataFrame de Pandas...")
    df = pd.DataFrame(processed_data)
    
    # Poner 'emotion' primero
    cols = ['emotion'] + [col for col in df.columns if col != 'emotion']
    df = df[cols]

    # Guardar como CSV
    csv_filename = 'ssi_custom_features.csv'
    df.to_csv(csv_filename, index=False, sep=',', quoting=1)
    print(f"CSV guardado en: {csv_filename}")
    
    # Guardar como XLSX
    xlsx_filename = 'ssi_custom_features.xlsx'
    df.to_excel(xlsx_filename, index=False, sheet_name='Datos')
    
    print(f"\n¡Proceso completado!")
    print(f"XLSX guardado en: {xlsx_filename}")
    print(f"Total de registros: {len(df)}")
    print("\nVista previa de los datos:")
    print(df.head())
    print(f"\nForma del dataset: {df.shape}")
    print(f"Columnas: {list(df.columns[:10])}...")  # Mostrar solo primeras 10
else:
    print("No se procesaron datos.")
