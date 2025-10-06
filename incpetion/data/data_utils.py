import matplotlib.pyplot as plt
import numpy as np

def denormalize(img_tensor):
    """
    Desnormaliza un tensor de imagen (que estaba en rango [-1,1]) a [0,1].
    """
    img = img_tensor.clone().detach().cpu()
    img = img * 0.5 + 0.5  # inversa de Normalize(mean=0.5, std=0.5)
    img = np.clip(img.numpy(), 0, 1)
    return img

def show_batch(dataloader , classes = ['airplane', 'bird', 'car', 'cat', 'deer', 
           'dog', 'horse', 'monkey', 'ship', 'truck'], n_images=8):
    """
    Muestra n_images de un batch del dataloader con sus etiquetas.

    Args:
        dataloader: DataLoader de PyTorch.
        n_images (int): Número de imágenes a mostrar.
    """
    images, labels = next(iter(dataloader))
    
    # calcular filas y columnas
    rows = n_images // 4 if n_images % 4 == 0 else (n_images // 4) + 1
    cols = min(4, n_images)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i in range(n_images):
        img = denormalize(images[i]).transpose(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(classes[labels[i]])
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()