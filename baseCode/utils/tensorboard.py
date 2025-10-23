import subprocess

def start_tensorboard(logdir='runs', port=6006):
    """Iniciar TensorBoard en un proceso separado"""
    try:
        # Comando para iniciar TensorBoard
        cmd = f"tensorboard --logdir={logdir} --port={port} --bind_all"
        
        print(f"🚀 Iniciando TensorBoard...")
        print(f"📁 Directorio de logs: {logdir}")
        print(f"🌐 URL: http://localhost:{port}")
        print(f"💡 Comando: {cmd}")
        print("⚠️  Para detener TensorBoard, interrumpe el kernel o cierra el terminal")
        
        # Iniciar en proceso separado (no bloqueante)
        process = subprocess.Popen(cmd, shell=True)
        return process
        
    except Exception as e:
        print(f"❌ Error al iniciar TensorBoard: {e}")
        print("💡 Intenta ejecutar manualmente en terminal: tensorboard --logdir=runs")
        return None
