import os
import time
import json
import socket
import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from datetime import datetime
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
from argparse import ArgumentParser as argparse

from utils.utils import write_json, read_list
from utils.training import epoch_time, initialize_log
from utils.csv_dataset_builder import build_dataset_from_csv, save_label_mapping
from dataloaders.data_augmentation import data_aug_selector
from utils.manual_stop import check_stop_training  # Para parada manual vía JSON
from test_classification import test_model
from models.classification import load_model
from dataloaders.Image_Dataset import Image_Dataset

def compute_acc(gt, pred):
    N = gt.shape[0]
    correct = gt == pred
    acc = correct.sum()/N
    return acc

def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, best_epoch, 
                   train_loss_history, train_acc_history, val_loss_history, 
                   val_acc_history, epochs, args, filepath):
    """Guarda un checkpoint completo del entrenamiento"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'train_loss_history': train_loss_history,
        'train_acc_history': train_acc_history,
        'val_loss_history': val_loss_history,
        'val_acc_history': val_acc_history,
        'epochs': epochs,
        'args': vars(args)
    }
    torch.save(checkpoint, filepath)

def train_loop(model, device, data_loader, criterion, optimizer):
    acc = []
    losses = []
    model.train()
    for images, labels, ind in tqdm(data_loader, desc="Train loop"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(images)

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        acc.append(compute_acc(labels, preds.argmax(1)).cpu().numpy())

    return np.mean(losses), np.mean(acc)

def validation_loop(model, device, data_loader, criterion):
    acc = []
    losses = []
    model.eval()
    with torch.no_grad():
        for images, labels, ind in tqdm(data_loader, desc="Val  loop "):
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)

            loss = criterion(preds, labels)

            losses.append(loss.detach().cpu().numpy())
            acc.append(compute_acc(labels, preds.argmax(1)).cpu().numpy())

    return np.mean(losses), np.mean(acc)


def main(args):
    # Set image name
    if 'ip' in socket.gethostname():
        pc_name = 'AWS'
    else:
        pc_name = socket.gethostname()
    timestamp  = datetime.today().strftime('%Y%m%d_%H%M%S')
    args.model_name = '{}_clas_{}_{}'.format(args.backbone,timestamp,pc_name)

    # Get image size
    img_size = (args.img_size, args.img_size)

    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Working with: {}".format(device))

    # Create output paths
    model_path = "./checkpoints/{}/".format(args.model_name)
    os.makedirs(model_path, exist_ok=True)
    model_save_path_best = os.path.join(model_path, "best_model.pth")
    model_save_path_last = os.path.join(model_path, "last_model.pth")
    json_log_path = os.path.join(model_path, "log.json")
    loss_fig_path = os.path.join(model_path, "loss.svg")

    # Print info
    print(" ")
    print("Model architecture:")
    summary(model, input_size=(3, args.img_size, args.img_size))
    print(" ")
    print("Dataset: {}".format(args.dataset))
    print("Train images: {:d}".format(len(train_dataset)))
    print("Validation images: {:d}".format(len(validation_dataset)))
    print("DA Library: {}".format(args.da_library))
    print("DA Level: {}".format(args.da_level))
    print(" ")
    print("Model name: {}".format(args.model_name))
    print("Backbone: {}".format(args.backbone))
    print("Weights: {}".format(args.weights))
    print("Image size: {}".format(img_size))
    print("N classes: {}".format(args.classes))
    print("Epochs: {:d}".format(args.epochs))
    print("bs: {:d}".format(args.batch_size))
    print("lr: {:f}".format(args.learning_rate))
    print("lr update freq: {:d}".format(args.lr_update_freq))
    print("jobs: {:d}".format(args.jobs))
    if resuming_from_checkpoint:
        print("Resuming from epoch: {:d}".format(start_epoch))
    print(" ")
    figure_title = args.model_name
    lists_path = os.path.join(model_path, 'lists')
    os.makedirs(lists_path, exist_ok=True)

    # =============================
    # Construcción de listas desde CSV (opcional)
    # Si se proporciona --csv_metadata se generarán train/validation/test automáticamente
    # =============================
    if getattr(args, 'csv_metadata', None):
        # Usar la utilidad para construir el dataset desde CSV
        dataset_info = build_dataset_from_csv(
            csv_metadata=args.csv_metadata,
            images_dir=args.images_dir,
            label_col=args.label_col,
            image_id_col=args.image_id_col,
            allowed_labels=args.allowed_labels,
            val_split=args.val_split,
            test_split=args.test_split,
            limit=args.limit,
            output_dir=lists_path,
            verbose=True
        )
        
        # Obtener rutas de las listas generadas
        train_list = dataset_info['train_list']
        validation_list = dataset_info['validation_list']
        test_list = dataset_info['test_list']
        
        # Actualizar configuración
        args.dataset = lists_path
        args.classes = dataset_info['num_classes']
        
        # Guardar mapeo de etiquetas
        mapping_path = os.path.join(model_path, 'label_mapping.json')
        save_label_mapping(dataset_info['label_mapping'], mapping_path)
    else:
        # Uso tradicional de listas ya existentes
        train_list = os.path.join(args.dataset, 'train.txt')
        validation_list = os.path.join(args.dataset, 'validation.txt')
        test_list = os.path.join(args.dataset, 'test.txt')

    # Transformaciones de entrenamiento (Data Augmentation)
    transform = data_aug_selector(args)

    print("\nLoading training set ...")
    train_dataset = Image_Dataset(train_list, args=args, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.jobs)

    print("\nLoading validation set ...")
    validation_dataset = Image_Dataset(validation_list, img_size=img_size, transform=None)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.jobs)

    # Get number of classes (si no se estableció antes por CSV)
    if not getattr(args, 'classes', None):
        args.classes = train_dataset.n_classes

    # Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.lr_update_freq:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_freq, gamma=0.1)
    else:
        scheduler = None

    # Cargar estados del optimizador y scheduler si se está reanudando
    if resuming_from_checkpoint and optimizer_state is not None:
        print("Cargando estados del optimizador y scheduler...")
        optimizer.load_state_dict(optimizer_state)
        if scheduler and scheduler_state:
            scheduler.load_state_dict(scheduler_state)
        
        # Actualizar nombre del modelo para evitar conflictos
        timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
        args.model_name = f'{args.backbone}_clas_resume_{timestamp}_{pc_name}'
        model_path = f"./checkpoints/{args.model_name}/"
        os.makedirs(model_path, exist_ok=True)
        model_save_path_best = os.path.join(model_path, "best_model.pth")
        model_save_path_last = os.path.join(model_path, "last_model.pth")
        json_log_path = os.path.join(model_path, "log.json")
        loss_fig_path = os.path.join(model_path, "loss.svg")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_freq, gamma=0.1)

    # Resume from checkpoint if specified
    start_epoch = 1
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    epochs = []
    best_loss = 1000
    best_epoch = 0
    resuming_from_checkpoint = False
    optimizer_state = None
    scheduler_state = None
    
    if args.resume and args.resume not in ["", "None", None, "none"]:
        if os.path.isfile(args.resume):
            print(f"\nCargando checkpoint desde: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Extraer directorio del checkpoint para buscar log.json
            checkpoint_dir = os.path.dirname(args.resume)
            log_json_path = os.path.join(checkpoint_dir, 'log.json')
            
            # Cargar configuración original desde log.json
            if os.path.isfile(log_json_path):
                print(f"Cargando configuración original desde: {log_json_path}")
                with open(log_json_path, 'r') as f:
                    original_config = json.load(f)
                
                # Sobrescribir parámetros críticos del checkpoint
                args.backbone = original_config.get('backbone', args.backbone)
                args.classes = original_config.get('classes', args.classes)
                args.img_size = original_config.get('image_size', args.img_size)
                
                print(f"✅ Configuración cargada desde checkpoint:")
                print(f"   Backbone: {args.backbone}")
                print(f"   Classes: {args.classes}")
                print(f"   Image size: {args.img_size}")
            else:
                print(f"⚠️ No se encontró log.json en {log_json_path}")
                print("   Usando configuración desde argumentos del checkpoint...")
                # Fallback: usar args guardados en el checkpoint si existen
                if 'args' in checkpoint:
                    saved_args = checkpoint['args']
                    args.backbone = saved_args.get('backbone', args.backbone)
                    args.classes = saved_args.get('classes', args.classes)
                    args.img_size = saved_args.get('img_size', args.img_size)
            
            # Ahora cargar el modelo con la configuración correcta
            model = load_model(args.backbone, args.weights, args.classes)
            model.to(device)
            
            # Cargar estado del modelo
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Cargar estado del optimizador (se hará después de crear el optimizador)
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint.get('scheduler_state_dict')
            
            # Restaurar variables de entrenamiento
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', 1000)
            best_epoch = checkpoint.get('best_epoch', 0)
            train_loss_history = checkpoint.get('train_loss_history', [])
            train_acc_history = checkpoint.get('train_acc_history', [])
            val_loss_history = checkpoint.get('val_loss_history', [])
            val_acc_history = checkpoint.get('val_acc_history', [])
            epochs = checkpoint.get('epochs', [])
            
            print(f"✅ Checkpoint cargado exitosamente")
            print(f"   Reanudando desde época: {start_epoch}")
            print(f"   Mejor época hasta ahora: {best_epoch} (val_loss: {best_loss:.5f})")
            print(f"   Historial de entrenamiento: {len(train_loss_history)} épocas")
            
            # Marcar que se está reanudando para cargar estados después
            resuming_from_checkpoint = True
        else:
            print(f"⚠️ Archivo de checkpoint no encontrado: {args.resume}")
            print("   Iniciando entrenamiento desde cero...")
            resuming_from_checkpoint = False
    else:
        resuming_from_checkpoint = False
    
    # Si no se está reanudando, crear modelo normalmente
    if not resuming_from_checkpoint:
        # Get pretrained model
        model = load_model(args.backbone, args.weights, args.classes)
        model.to(device)

        # Load initial weights (not for resuming, only for pretrained weights)
        if args.weights not in ["", "None", None, "none", "imagenet"]:
            print(f"\nCargando pesos preentrenados desde: {args.weights}")
            model.load_state_dict(torch.load(args.weights, map_location=device))

    # TensorBoard writer
    if getattr(args, 'tensorboard', False):
        # Guardar logs de TensorBoard dentro del directorio del modelo
        tb_log_dir = os.path.join(model_path, 'tensorboard_logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(tb_log_dir)
        print(f"TensorBoard activado. Log dir: {tb_log_dir}")
        print(f"Para visualizar: tensorboard --logdir={model_path}")
    else:
        writer = None

    # Loop variables
    log_dict = initialize_log(args, type='classification')
    log_dict["training_images"] = len(train_dataset)
    log_dict["validation_images"] = len(validation_dataset)
    epoch_dt = []
    control_file_path = os.path.join('assets', 'training_control.json')  # Ruta del archivo de control
    log_dict["early_stop"] = False
    log_dict["stop_epoch"] = None

    # Train model
    T0 = time.time()
    for e in range(start_epoch, args.epochs+1):
        print('\nepoch : {:d}'.format(e))
        epochs.append(e)
        t0 = time.time()

        # Train loop
        train_loss, train_acc = train_loop(model, device, train_loader, criterion, optimizer)

        # Validation loop
        val_loss, val_acc = validation_loop(model, device, validation_loader, criterion)

        # Update scheduler
        if args.lr_update_freq:
            scheduler.step()

        # Store los values
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        # Stop timer
        t1 = time.time()
        epoch_dt.append(t1-t0)

        # Print epoch info
        print('training  : loss={:.5f} , acc={:0.3%}'.format(train_loss, train_acc))
        print('validation: loss={:.5f} , acc={:0.3%}'.format(val_loss, val_acc))
        print(epoch_time(t0, t1))

        # TensorBoard scalars
        if writer:
            writer.add_scalar('Train/Loss', train_loss, e)
            writer.add_scalar('Train/Accuracy', train_acc, e)
            writer.add_scalar('Val/Loss', val_loss, e)
            writer.add_scalar('Val/Accuracy', val_acc, e)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], e)
            if e % 5 == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(f'Params/{name}', param, e)
                        if param.grad is not None:
                            writer.add_histogram(f'Grads/{name}', param.grad, e)

        # Save latest model
        save_checkpoint(model, optimizer, scheduler if args.lr_update_freq else None, 
                       e, best_loss, best_epoch, train_loss_history, train_acc_history,
                       val_loss_history, val_acc_history, epochs, args, model_save_path_last)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = e
            save_checkpoint(model, optimizer, scheduler if args.lr_update_freq else None, 
                           e, best_loss, best_epoch, train_loss_history, train_acc_history,
                           val_loss_history, val_acc_history, epochs, args, model_save_path_best)

        # Plot loss
        fig, ax = plt.subplots(1,1, figsize=(8,5))
        ax.plot(epochs, train_loss_history, label='training loss')
        ax.plot(epochs, val_loss_history, label='validation loss')
        ax.plot(epochs, train_acc_history, label='training acc')
        ax.plot(epochs, val_acc_history, label='validation acc')
        ax.set_title(figure_title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
        ax.legend()
        plt.savefig(loss_fig_path, format="svg")

        # Measure total training time
        T1 = time.time()

        # Update log_dict
        log_dict["epoch"] = e
        log_dict["val_loss"] = float(val_loss)
        log_dict["best_epoch"] = best_epoch
        log_dict["best_val_loss"] = float(best_loss)
        log_dict["Training_Time"] = epoch_time(T0, T1)
        log_dict["Avg_Epoch_Time"] = epoch_time(0, np.mean(epoch_dt))
        write_json(log_dict, json_log_path)

        # Revisión de parada manual después de guardar el log/modelo
        try:
            if check_stop_training(control_file_path):
                print(f"\nParada manual solicitada. Deteniendo entrenamiento tras la época {e}.")
                log_dict["early_stop"] = True
                log_dict["stop_epoch"] = e
                write_json(log_dict, json_log_path)  # Actualizar log con información de parada
                if writer:
                    writer.add_text('Training/Status', f'Early stop at epoch {e}')
                break
        except Exception as ex:
            print(f"No se pudo leer la bandera de parada: {ex}")


    # Load best model
    print("\nLoading best model")
    print("Epoch: {:d}".format(best_epoch))
    print("Path: {}".format(model_save_path_best))
    
    # Manejar tanto formato nuevo (checkpoint) como viejo (solo state_dict)
    try:
        checkpoint = torch.load(model_save_path_best, map_location=device)
        if 'model_state_dict' in checkpoint:
            # Formato nuevo (checkpoint completo)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Formato viejo (solo state_dict)
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error cargando el mejor modelo: {e}")
        print("Usando el modelo actual...")
    
    model.eval()
    
    # Evaluate model on test set
    print("\nEvaluating on test set ...")
    y_true, y_pred = test_model(test_list, model_path, args.batch_size, args.jobs, model)

    # Compute test accuracy
    acc = sum(y_true == y_pred) / len(y_pred)
    print("Test set accuracy: {:0.4f}".format(acc))
    print(" ")

    if writer:
        writer.add_scalar('Test/Accuracy', float(acc))
        writer.add_text('Model/Info', f'Best epoch: {best_epoch}, Early stop: {log_dict["early_stop"]}')
        writer.close()
        print("TensorBoard writer cerrado.")
    

if __name__ == '__main__':
    parser = argparse()
    parser.add_argument('-d',   '--dataset',                                type=str,       help='Path to the lists of the dataset.')
    parser.add_argument('-b',   '--backbone',       default="vgg16",        type=str,       help='Conv-Net backbone.')
    parser.add_argument('-w',   '--weights',                                type=str,       help="Model's initial Weights: < none | imagenet | /path/to/weights/ >")
    parser.add_argument('-sz',  '--img_size',       default=224,            type=int,       help='Image size.')
    parser.add_argument('-e',   '--epochs',         default=2,              type=int,       help='Number of epochs.')
    parser.add_argument('-bs',  '--batch_size',     default=32,             type=int,       help='Batch size.')
    parser.add_argument('-j',   '--jobs',           default=8,              type=int,       help="Number of workers for dataloader's parallel jobs.")
    parser.add_argument('-lr',  '--learning_rate',  default=0.0001,         type=float,     help='Learning Rate.')
    parser.add_argument('-lrf', '--lr_update_freq', default=0,              type=int,       help='Learning rate update frequency in epochs.')
    parser.add_argument('-da',  '--da_library',     default="torchvision",  type=str,       help='Data Augmentation library: < albumentations | torchvision >')
    parser.add_argument('-lvl', '--da_level',       default="heavy",        type=str,       help='Data Augmentation level: < light | medium | heavy >')
    parser.add_argument('-csv', '--csv_metadata',   default="../DataTIC/bcn20000_metadata_2025-10-19.csv",             type=str,       help='Ruta al CSV con metadatos para generar listas.')
    parser.add_argument('-img', '--images_dir',     default="../DataTIC/ISIC-images",                        type=str,       help='Directorio raíz de las imágenes referenciadas en el CSV.')
    parser.add_argument('-l',   '--label_col',      default='diagnosis_1',  type=str,       help='Nombre de la columna de la etiqueta en el CSV.')
    parser.add_argument('-id',  '--image_id_col',   default='isic_id',      type=str,       help='Columna que contiene el ID base de la imagen.')
    parser.add_argument('-al',  '--allowed_labels', default="Benign,Malignant",              type=str,       help='Lista separada por comas de labels permitidos (opcional).')
    parser.add_argument('-vs',  '--val_split',      default=0.2,            type=float,     help='Proporción de validación al generar listas desde CSV.')
    parser.add_argument('-ts',  '--test_split',     default=0.0,            type=float,     help='Proporción de test al generar listas desde CSV.')
    parser.add_argument('-lim', '--limit',          default=0,              type=int,       help='Límite de imágenes a usar desde el CSV (0 = sin límite).')
    parser.add_argument('-res', '--resume',                                 type=str,       help='Ruta al checkpoint para reanudar entrenamiento.')
    parser.add_argument('-tb',  '--tensorboard',    action='store_true',                    help='Log training metrics to TensorBoard.')
    args = parser.parse_args()

    main(args)
