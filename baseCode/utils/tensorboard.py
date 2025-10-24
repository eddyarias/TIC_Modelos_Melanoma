import subprocess

def start_tensorboard(logdir='runs', port=6006):
    """Iniciar TensorBoard en un proceso separado"""
    try:
        cmd = f"tensorboard --logdir={logdir} --port={port} --bind_all"
        
        print(f"Iniciando TensorBoard...")
        print(f"Directorio de logs: {logdir}")
        print(f"URL: http://localhost:{port}")
        print(f"Comando: {cmd}")
        process = subprocess.Popen(cmd, shell=True)
        return process
        
    except Exception as e:
        print(f"Error al iniciar TensorBoard: {e}")
        return None
