import os
import time
import json
import socket
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser as argparse

from utils.utils import write_json, read_list
from utils.training import epoch_time, initialize_log
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
    # Visdom Visualization
    if args.visdom:
        print('Initializing Visdom')
        import visdom
        from utils.linePlotter import VisdomLinePlotter
        vis = visdom.Visdom()
        plotter = VisdomLinePlotter()
    else:
        vis = None

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
        print("\nGenerando listas desde CSV:")
        if not os.path.isfile(args.csv_metadata):
            raise FileNotFoundError(f"CSV no encontrado: {args.csv_metadata}")
        df = pd.read_csv(args.csv_metadata)
        label_col = args.label_col
        if label_col not in df.columns:
            raise ValueError(f"Columna '{label_col}' no encontrada en CSV")
        # Filtrar labels permitidos si se definieron
        if args.allowed_labels:
            df = df[df[label_col].isin(args.allowed_labels.split(','))]
        else:
            # Por defecto usar sólo Benign/Malignant si existen
            if set(['Benign','Malignant']).issubset(set(df[label_col].dropna().unique())):
                df = df[df[label_col].isin(['Benign','Malignant'])]
        # Limitar
        if args.limit > 0:
            df = df.head(args.limit)
        # Generar filepath
        if 'isic_id' in df.columns and args.image_id_col == 'isic_id':
            df['filename'] = df[args.image_id_col].astype(str) + '.jpg'
        else:
            df['filename'] = df[args.image_id_col].astype(str)
        df['filepath'] = df['filename'].apply(lambda x: os.path.join(args.images_dir, x))
        # Filtrar por existencia
        df = df[df['filepath'].apply(os.path.exists)]
        if len(df) == 0:
            raise ValueError("No se encontraron imágenes existentes tras el filtrado.")
        # Map labels
        unique_labels = sorted(df[label_col].dropna().unique())
        label2idx = {l:i for i,l in enumerate(unique_labels)}
        print(f"Labels detectados: {label2idx}")
        df['label_idx'] = df[label_col].apply(lambda x: label2idx[x])
        # Splits
        test_split = max(0.0, min(0.9, args.test_split))
        val_split = max(0.0, min(0.9, args.val_split))
        if test_split + val_split >= 0.95:
            raise ValueError("La suma de val_split y test_split es demasiado alta")
        remaining_df = df
        # Primero split test si corresponde
        if test_split > 0:
            remaining_df, test_df = train_test_split(remaining_df, test_size=test_split, random_state=42, stratify=remaining_df['label_idx'])
        else:
            test_df = pd.DataFrame(columns=remaining_df.columns)
        # Split validation
        if val_split > 0:
            train_df, val_df = train_test_split(remaining_df, test_size=val_split, random_state=42, stratify=remaining_df['label_idx'])
        else:
            train_df = remaining_df
            val_df = pd.DataFrame(columns=remaining_df.columns)
        print(f"Split -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        # Escribir listas
        def write_list(df_part, name):
            list_path = os.path.join(lists_path, f"{name}.txt")
            lines = [f"{row.filepath} {int(row.label_idx)}" for _, row in df_part.iterrows()]
            with open(list_path, 'w') as f:
                f.write('\n'.join(lines))
            return list_path
        train_list = write_list(train_df, 'train')
        validation_list = write_list(val_df, 'validation') if len(val_df) else write_list(train_df.sample(min(1,len(train_df))), 'validation')  # fallback mínimo
        test_list = write_list(test_df, 'test') if len(test_df) else write_list(train_df.sample(min(1,len(train_df))), 'test')
        # Actualizar args.dataset a nueva ruta para consistencia
        args.dataset = lists_path
        # Ajustar número de clases
        args.classes = len(label2idx)
        # Guardar mapping
        mapping_path = os.path.join(model_path, 'label_mapping.json')
        write_json(label2idx, mapping_path)
        print(f"Mapeo de labels guardado en {mapping_path}")
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

        # Visualize on Visdom
        if args.visdom:
            plotter.plot('value', 'training  loss ', figure_title, e, train_loss)
            plotter.plot('value', 'validation loss', figure_title, e, val_loss)
            plotter.plot('value', 'training  acc ', figure_title, e, train_acc)
            plotter.plot('value', 'validation acc', figure_title, e, val_acc)

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
                print(f"\n🛑 Parada manual solicitada. Deteniendo entrenamiento tras la época {e}.")
                log_dict["early_stop"] = True
                log_dict["stop_epoch"] = e
                write_json(log_dict, json_log_path)  # Actualizar log con información de parada
                if writer:
                    writer.add_text('Training/Status', f'Early stop at epoch {e}')
                break
        except Exception as ex:
            print(f"⚠️ No se pudo leer la bandera de parada: {ex}")


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
        print(f"⚠️ Error cargando el mejor modelo: {e}")
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
    parser.add_argument('-d', '--dataset', type=str,
                        help='Path to the lists of the dataset.')
    parser.add_argument('-b', '--backbone', type=str, default="vgg16",
                        help='Conv-Net backbone.')
    parser.add_argument('-w', '--weights', type=str, default="",
                        help="Model's initial Weights: < none | imagenet | /path/to/weights/ >")
    parser.add_argument('-sz', '--img_size', type=int, default=224,
                        help='Image size.')
    parser.add_argument('-e', '--epochs', type=int, default=2,
                        help='Number of epochs.')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('-j', '--jobs', type=int, default=8,
                        help="Number of workers for dataloader's parallel jobs.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                        help='Learning Rate.')
    parser.add_argument('-lrf', '--lr_update_freq', type=int, default=0,
                        help='Learning rate update frequency in epochs.')
    parser.add_argument('-da', '--da_library', type=str, default="torchvision",
                        help='Data Augmentation library: < albumentations | torchvision >')
    parser.add_argument('-lvl', '--da_level', type=str, default="heavy",
                        help='Data Augmentation level: < light | medium | heavy >')
    parser.add_argument('-vis', '--visdom', action='store_true',
                        help='Visualize training on visdom.')
    parser.add_argument('-tb', '--tensorboard', action='store_true',
                        help='Log training metrics to TensorBoard.')
    parser.add_argument('--csv_metadata', type=str, default='', help='Ruta al CSV con metadatos para generar listas.')
    parser.add_argument('--images_dir', type=str, default='', help='Directorio raíz de las imágenes referenciadas en el CSV.')
    parser.add_argument('--label_col', type=str, default='diagnosis_1', help='Nombre de la columna de la etiqueta en el CSV.')
    parser.add_argument('--image_id_col', type=str, default='isic_id', help='Columna que contiene el ID base de la imagen.')
    parser.add_argument('--allowed_labels', type=str, default='', help='Lista separada por comas de labels permitidos (opcional).')
    parser.add_argument('--val_split', type=float, default=0.2, help='Proporción de validación al generar listas desde CSV.')
    parser.add_argument('--test_split', type=float, default=0.0, help='Proporción de test al generar listas desde CSV.')
    parser.add_argument('--limit', type=int, default=0, help='Límite de imágenes a usar desde el CSV (0 = sin límite).')
    parser.add_argument('--resume', type=str, default='', help='Ruta al checkpoint para reanudar entrenamiento.')
    args = parser.parse_args()

    main(args)
