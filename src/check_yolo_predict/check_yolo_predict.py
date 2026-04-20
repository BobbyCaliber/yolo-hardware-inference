from ultralytics import YOLO
import torch
import cpuinfo
import GPUtil
import psutil
import numpy as np
import random
import os
import time
import csv

folder_path = os.environ.get('IMAGES_DIR', '/app/data/images')
output_dir = os.environ.get('DATA_DIR', '/app/data')
os.makedirs(output_dir, exist_ok=True)


def count_parameters(model):
    model = model.model
    param_dict = dict()
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        param_dict[name] = params
        total_params += params
    return param_dict, total_params


def get_random_images_from_folder(folder_path, num_images):
    # Получаем список всех файлов в папке
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Проверяем, достаточно ли файлов для выборки
    if num_images > len(all_files):
        raise ValueError(f"Запрошено {num_images} изображений, но в папке только {len(all_files)} доступных файлов.")

    # Выбираем случайные изображения
    selected_files = random.sample(all_files, num_images)
    return selected_files


# константы
ram = int(np.round(psutil.virtual_memory().total / (1024. ** 3)))
cpu = cpuinfo.get_cpu_info().get('brand_raw', 'Unknown')

_gpus = GPUtil.getGPUs()
if _gpus:
    gpu = _gpus[0].name
    gpu_mem = _gpus[0].memoryTotal / 1024
else:
    gpu = 'none'
    gpu_mem = 0


devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda') # в train device=cuda

# model_names7 yolov7 not supported in ultralytics
model_names5 = ['yolov5nu.pt', 'yolov5su.pt', 'yolov5mu.pt', 'yolov5lu.pt']
model_names6 = ['yolov6n.yaml', 'yolov6s.yaml', 'yolov6m.yaml', 'yolov6l.yaml']
model_names8 = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']
model_names9 = ['yolov9t.pt', 'yolov9s.pt', 'yolov9m.pt', 'yolov9c.pt']
model_names10 = ['yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10l.pt']
model_names11 = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt']

models = [model_names5, model_names6, model_names8, model_names9, model_names10, model_names11]

img_sizes = [160, 320, 640, 800, 960, 1120]
img_batch_sizes = [1, 2, 5, 8, 10]

# inference on
with open(os.path.join(output_dir, 'yolo_predict.csv'), 'w', newline='', buffering=1) as file:  # line-buffered so partial results survive SIGTERM/SIGKILL
    writer = csv.writer(file)
    field = ['cpu_name',
             'gpu_name',
             'gpu_memory',
             'used_gpu',
             'ram_memory',
             'model_name',
             'model_param_dict',
             'num_all_params',
             'model_img_size',
             'image_batch_size',
             'predicting_times'
             ]
    writer.writerow(field)

    # определим константы
    cpu_name = cpu
    gpu_name = gpu
    gpu_memory = gpu_mem
    ram_memory = ram

    # варьируемые переменные
    for device in devices:
        if device == 'cpu':
            used_gpu = False
        elif device == 'cuda':
            used_gpu = True

        for model_group in models:
            for a in model_group:
                model_name = a
                model = YOLO(model_name)
                model_param_dict, num_all_params = count_parameters(model)
                for k in img_batch_sizes:
                    image_batch_size = k
                    for j in img_sizes:
                        model_img_size = j
                        random_images = get_random_images_from_folder(folder_path, image_batch_size)
                        since = time.time()
                        model.predict(
                            source=random_images,
                            imgsz=model_img_size,
                            device=device,
                            save=False
                        )
                        predicting_times = time.time() - since
                        row = [cpu_name,
                               gpu_name,
                               gpu_memory,
                               used_gpu,
                               ram_memory,
                               model_name,
                               model_param_dict,
                               num_all_params,
                               model_img_size,
                               image_batch_size,
                               predicting_times]
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                        writer.writerow(row)