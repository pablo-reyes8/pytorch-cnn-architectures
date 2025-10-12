import torch.nn as nn
import torch
import math


def get_same_pad(kernel_size: int, stride: int, dilation: int = 1):
    # para kernels impares típicos (3,5), same padding = floor((k-1)/2) cuando stride=1
    # para stride>1, replicamos el "same" de TF con ZeroPad2d dinámico (ver ConvBNAct abajo)
    pad = (kernel_size - 1) // 2 * dilation
    return pad

def calc_pad_2d(x: torch.Tensor, kernel_size: int, stride: int, dilation: int = 1):
    # calcula padding (top,bottom,left,right) para emular 'same' al downsamplear
    h, w = x.shape[-2:]
    out_h = math.ceil(h / stride)
    out_w = math.ceil(w / stride)
    pad_h = max((out_h - 1) * stride + (kernel_size - 1) * dilation + 1 - h, 0)
    pad_w = max((out_w - 1) * stride + (kernel_size - 1) * dilation + 1 - w, 0)

    # repartir padding en ambos lados
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)


def round_filters(c_out, width_mult, divisor = 8, min_width = None):
    """
    Redondea canales al múltiplo de 'divisor' (como EfficientNet).
    Se asegura de no reducir más de un 10% respecto a c_out * width_mult.
    """
    if min_width is None:
        min_width = divisor
    new_c = c_out * width_mult
    new_c = max(min_width, int(new_c + divisor / 2) // divisor * divisor)
    # Evita caer demasiado por redondeo:
    if new_c < 0.9 * c_out * width_mult:
        new_c += divisor
    return int(new_c)

def round_repeats(n, depth_mult):
    """
    Redondea repeticiones al entero hacia arriba.
    """
    return int(math.ceil(n * depth_mult))


def round_resolution(base_size, res_mult: float, divisor = 8, clamp = None):
    """
    Escala resolución base y redondea al múltiplo de 'divisor'.
    Opcionalmente, limita el rango con 'clamp=(min,max)'.
    """
    size = int(round(base_size * res_mult))
    size = max(divisor, int(round(size / divisor) * divisor))
    if clamp is not None:
        lo, hi = clamp
        size = max(lo, min(size, hi))
    return size


def make_divisible(v: float, divisor: int, min_value = None):
    """
    Redondea 'v' al múltiplo de 'divisor' sin caer >10% por debajo.
    (Estilo MobileNet/EfficientNet).
    """
    if divisor <= 0:
        return max(1, int(v))
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)