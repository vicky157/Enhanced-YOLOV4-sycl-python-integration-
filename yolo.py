import time
import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation

    def forward(self, x):
        start_time = time.time()  # Start timer
        x = self.conv(x)
        if self.activation == 'mish':
            x = torch.nn.functional.mish(x)
        elif self.activation == 'leaky':
            x = torch.nn.functional.leaky_relu(x)
        end_time = time.time()  # End timer
        return x, end_time - start_time

class YOLOv4(nn.Module):
    def __init__(self, config):
        super(YOLOv4, self).__init__()
        self.module_list = nn.ModuleList()
        in_channels = 3  # Initial channel (RGB)

        for layer in config:
            if layer['type'] == 'convolutional':
                out_channels = int(layer['filters'])
                kernel_size = int(layer['size'])
                stride = int(layer['stride'])
                padding = int(layer['pad'])
                activation = layer['activation']
                conv_layer = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation)
                self.module_list.append(conv_layer)
                in_channels = out_channels

    def forward(self, x):
        layer_times = []
        for module in self.module_list:
            x, time_taken = module(x)
            layer_times.append(time_taken)
        return x, layer_times

    
    
    
    
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


# Parse the YOLOv4 configuration
cfg_file_path = './yolov4.cfg'
yolov4_layers = parse_cfg(cfg_file_path)

# Create the YOLOv4 model
model = YOLOv4(yolov4_layers)

# Dummy input for testing
input_tensor = torch.randn(1, 3, 416, 416)

# Forward pass
output, layer_times = model(input_tensor)

# Print the time taken for each layer
for i, time_taken in enumerate(layer_times):
    print(f"Time in layer {i+1}: {time_taken:.6f} seconds")
