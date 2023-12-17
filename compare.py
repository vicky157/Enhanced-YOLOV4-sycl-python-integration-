import torch
from yolo import YOLOv4 as ModifiedYOLOv4
from yoto import YOLOv4 as OriginalYOLOv4
import time
import torch.nn as nn
def parse_cfg(cfg_file):
    """
    Parses the YOLO configuration file to extract layer information.
    """
    with open(cfg_file, 'r') as file:
        lines = file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    layer_configs = []
    layer_config = {}
    
    for line in lines:
        if line.startswith('['):  # This marks the start of a new layer
            if layer_config:
                layer_configs.append(layer_config)  # Add the previous layer's config
                layer_config = {}
            layer_type = line.strip('[]')
            layer_config['type'] = layer_type
        else:
            key, value = line.split('=')
            layer_config[key.rstrip()] = value.lstrip()

    if layer_config:
        layer_configs.append(layer_config)

    return layer_configs

def benchmark_model(model, input_tensor):
    start_time = time.time()
    with torch.no_grad():
        model(input_tensor)
    end_time = time.time()
    return end_time - start_time

# Load YOLOv4 configuration
cfg_file_path = './yolov4.cfg'
yolov4_layers = parse_cfg(cfg_file_path)

# Create models
original_model = OriginalYOLOv4(yolov4_layers)
modified_model = ModifiedYOLOv4(yolov4_layers)

# Dummy input for testing
input_tensor = torch.randn(1, 3, 416, 416)

# Benchmark the models
time_original = benchmark_model(original_model, input_tensor)
time_modified = benchmark_model(modified_model, input_tensor)

# Calculate and display the results
print(f"Original Model Time: {time_original:.6f} seconds")
print(f"Modified Model Time: {time_modified:.6f} seconds")
speedup = time_original / time_modified
print(f"Speedup: {speedup:}x")




