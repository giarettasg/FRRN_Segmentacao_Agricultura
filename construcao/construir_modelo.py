import sys, os
import tensorflow as tf
import subprocess

sys.path.append("modelo")
from modelo.FRRN import build_frrn


def build_model(model_name, net_input, num_classes, crop_width, crop_height, frontend="ResNet101", is_training=True):
    print("Inicializando modelo")

    network = None
    init_fn = None
    if model_name == "FRRN-A":
        network = build_frrn(net_input, preset_model=model_name, num_classes=num_classes)

    return network, init_fn