# Definir función de entrenamiento con tqdm simplificado, control de parada y TensorBoard
import json

def check_stop_training(control_file_path):
    """Revisar el archivo JSON para verificar si se debe parar el entrenamiento"""
    try:
        with open(control_file_path, 'r') as f:
            control_data = json.load(f)
        return control_data.get('stop_training', False)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    

    # Funciones auxiliares para controlar el entrenamiento
def set_stop_training(stop=True):
    """Establecer la bandera de parada del entrenamiento"""
    control_file_path = 'baseCode/assets/training_control.json'
    try:
        with open(control_file_path, 'r') as f:
            control_data = json.load(f)
        
        control_data['stop_training'] = stop
        
        with open(control_file_path, 'w') as f:
            json.dump(control_data, f, indent=4)
        
        action = "activada" if stop else "desactivada"
        print(f"✅ Bandera de parada {action} exitosamente")
        
    except Exception as e:
        print(f"❌ Error al modificar el archivo de control: {e}")

def get_training_status():
    """Obtener el estado actual del control de entrenamiento"""
    control_file_path = 'baseCode/assets/training_control.json'
    try:
        with open(control_file_path, 'r') as f:
            control_data = json.load(f)
        
        stop_flag = control_data.get('stop_training', False)
        status = "🛑 PARAR" if stop_flag else "▶️ CONTINUAR"
        print(f"Estado actual: {status}")
        return stop_flag
        
    except Exception as e:
        print(f"❌ Error al leer el archivo de control: {e}")
        return False
