"""
Utilidades para construir listas de entrenamiento/validación/test desde archivos CSV
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils import write_json


def build_dataset_from_csv(csv_metadata, images_dir, label_col='diagnosis_1', 
                          image_id_col='isic_id', allowed_labels=None, 
                          val_split=0.2, test_split=0.0, limit=0, 
                          output_dir=None, verbose=True):
    """
    Construye listas de entrenamiento/validación/test desde un archivo CSV.
    
    Args:
        csv_metadata (str): Ruta al archivo CSV con metadatos
        images_dir (str): Directorio raíz donde están las imágenes
        label_col (str): Nombre de la columna que contiene las etiquetas
        image_id_col (str): Nombre de la columna que contiene el ID de la imagen
        allowed_labels (list or str): Lista o string separado por comas de etiquetas permitidas
        val_split (float): Proporción para validación (0.0-0.9)
        test_split (float): Proporción para test (0.0-0.9)
        limit (int): Límite de imágenes a usar (0 = sin límite)
        output_dir (str): Directorio donde guardar las listas
        verbose (bool): Si mostrar información detallada
        
    Returns:
        dict: Diccionario con rutas de las listas y metadatos
        {
            'train_list': ruta_archivo_train,
            'validation_list': ruta_archivo_val,
            'test_list': ruta_archivo_test,
            'label_mapping': mapeo_etiqueta_a_indice,
            'num_classes': numero_de_clases,
            'stats': estadisticas_del_split
        }
    """
    if verbose:
        print("\nGenerando listas desde CSV:")
    
    # Validar archivo CSV
    if not os.path.isfile(csv_metadata):
        raise FileNotFoundError(f"CSV no encontrado: {csv_metadata}")
    
    # Cargar CSV
    df = pd.read_csv(csv_metadata)
    
    # Validar columnas
    if label_col not in df.columns:
        raise ValueError(f"Columna '{label_col}' no encontrada en CSV")
    if image_id_col not in df.columns:
        raise ValueError(f"Columna '{image_id_col}' no encontrada en CSV")
    
    if verbose:
        print(f"CSV cargado: {len(df)} filas")
        print(f"Columna de etiquetas: {label_col}")
        print(f"Columna de ID imagen: {image_id_col}")
    
    # Filtrar etiquetas permitidas
    if allowed_labels:
        if isinstance(allowed_labels, str):
            allowed_labels = allowed_labels.split(',')
        allowed_labels = [label.strip() for label in allowed_labels]
        df = df[df[label_col].isin(allowed_labels)]
        if verbose:
            print(f"Filtrado por etiquetas permitidas {allowed_labels}: {len(df)} filas")
    else:
        # Por defecto usar solo Benign/Malignant si existen
        available_labels = set(df[label_col].dropna().unique())
        if set(['Benign', 'Malignant']).issubset(available_labels):
            df = df[df[label_col].isin(['Benign', 'Malignant'])]
            if verbose:
                print(f"Filtrado automático por Benign/Malignant: {len(df)} filas")
    
    # Limitar número de muestras
    if limit > 0:
        df = df.head(limit)
        if verbose:
            print(f"Limitado a {limit} muestras: {len(df)} filas")
    
    # Generar rutas de archivos
    if 'isic_id' in df.columns and image_id_col == 'isic_id':
        df['filename'] = df[image_id_col].astype(str) + '.jpg'
    else:
        df['filename'] = df[image_id_col].astype(str)
    
    df['filepath'] = df['filename'].apply(lambda x: os.path.join(images_dir, x))
    
    # Filtrar por existencia de archivos
    existing_mask = df['filepath'].apply(os.path.exists)
    missing_files = len(df) - existing_mask.sum()
    df = df[existing_mask]
    
    if verbose and missing_files > 0:
        print(f"⚠️ {missing_files} archivos no encontrados, quedan {len(df)} válidos")
    
    if len(df) == 0:
        raise ValueError("No se encontraron imágenes existentes tras el filtrado.")
    
    # Mapear etiquetas a índices
    unique_labels = sorted(df[label_col].dropna().unique())
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    if verbose:
        print(f"Labels detectados: {label2idx}")
    
    df['label_idx'] = df[label_col].apply(lambda x: label2idx[x])
    
    # Validar splits
    test_split = max(0.0, min(0.9, test_split))
    val_split = max(0.0, min(0.9, val_split))
    
    if test_split + val_split >= 0.95:
        raise ValueError("La suma de val_split y test_split es demasiado alta (>= 0.95)")
    
    # Realizar splits estratificados
    remaining_df = df
    
    # Split test si corresponde
    if test_split > 0:
        remaining_df, test_df = train_test_split(
            remaining_df, 
            test_size=test_split, 
            random_state=42, 
            stratify=remaining_df['label_idx']
        )
    else:
        test_df = pd.DataFrame(columns=remaining_df.columns)
    
    # Split validation
    if val_split > 0:
        train_df, val_df = train_test_split(
            remaining_df, 
            test_size=val_split, 
            random_state=42, 
            stratify=remaining_df['label_idx']
        )
    else:
        train_df = remaining_df
        val_df = pd.DataFrame(columns=remaining_df.columns)
    
    if verbose:
        print(f"Split -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Escribir listas si se especifica directorio de salida
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        def write_list(df_part, name):
            list_path = os.path.join(output_dir, f"{name}.txt")
            lines = [f"{row.filepath} {int(row.label_idx)}" for _, row in df_part.iterrows()]
            with open(list_path, 'w') as f:
                f.write('\n'.join(lines))
            return list_path
        
        train_list = write_list(train_df, 'train')
        
        # Validation list (con fallback mínimo)
        if len(val_df) > 0:
            validation_list = write_list(val_df, 'validation')
        else:
            # Fallback: usar una muestra mínima del training
            fallback_val = train_df.sample(min(1, len(train_df)))
            validation_list = write_list(fallback_val, 'validation')
        
        # Test list (con fallback mínimo)
        if len(test_df) > 0:
            test_list = write_list(test_df, 'test')
        else:
            # Fallback: usar una muestra mínima del training
            fallback_test = train_df.sample(min(1, len(train_df)))
            test_list = write_list(fallback_test, 'test')
        
        if verbose:
            print(f"Listas guardadas en: {output_dir}")
    else:
        train_list = None
        validation_list = None
        test_list = None
    
    # Estadísticas del split
    stats = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'label_distribution': df[label_col].value_counts().to_dict(),
        'missing_files': missing_files
    }
    
    return {
        'train_list': train_list,
        'validation_list': validation_list,
        'test_list': test_list,
        'label_mapping': label2idx,
        'num_classes': len(label2idx),
        'stats': stats,
        'dataframes': {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
    }


def save_label_mapping(label_mapping, output_path):
    """
    Guarda el mapeo de etiquetas a archivo JSON.
    
    Args:
        label_mapping (dict): Mapeo de etiqueta a índice
        output_path (str): Ruta donde guardar el archivo JSON
    """
    write_json(label_mapping, output_path)
    print(f"Mapeo de labels guardado en {output_path}")


def load_label_mapping(json_path):
    """
    Carga el mapeo de etiquetas desde archivo JSON.
    
    Args:
        json_path (str): Ruta al archivo JSON
        
    Returns:
        dict: Mapeo de etiqueta a índice
    """
    import json
    with open(json_path, 'r') as f:
        return json.load(f)