from torchvision import transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Dict
import torch
import torchvision as tv
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split


ROOT = "./data"     # carpeta donde se descargará el dataset
IMG_SIZE = (256, 256)

# --- Transforms ---
img_tf = T.Compose([
    T.Resize(IMG_SIZE, interpolation=Image.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),])


# Para visualizar, también quiero la imagen "no normalizada":
img_tf_unnorm = T.Resize(IMG_SIZE, interpolation=Image.BILINEAR)

mask_resize = T.Resize(IMG_SIZE, interpolation=Image.NEAREST)

def mask_decode_to_rgb(mask_np):
    """
    Oxford-IIIT Pet (segmentation trimaps):
      1 = fondo, 2 = borde, 3 = mascota
    Devuelve una imagen RGB colorizada para visualizar.
    """
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Colores (BGR/BRG? No, usamos RGB)
    # fondo: negro, borde: amarillo, mascota: verde
    rgb[mask_np == 1] = (0, 0, 0)
    rgb[mask_np == 2] = (255, 255, 0)
    rgb[mask_np == 3] = (0, 200, 0)
    return rgb

class OxfordPetsSeg(Dataset):
    """
    Dataset para Oxford-IIIT Pet con segmentación.
    - binary=True: devuelve máscara binaria (fondo=0, {borde|mascota}=1)
    - binary=False: devuelve máscara multiclase remapeada a {0,1,2}={fondo,borde,mascota}
    Augment opcional (flip horizontal coherente imagen/máscara).
    """
    def __init__(self,root: str = "./data",
                 split: str = "trainval",img_size: Tuple[int,int] = (256,256),
                 binary: bool = True,augment: bool = False,download: bool = True):

        assert split in {"trainval", "test"}

        self.base = tv.datasets.OxfordIIITPet(
            root=root, split=split, target_types="segmentation", download=download)


        self.img_size = img_size
        self.binary = binary
        self.augment = augment

        self.img_resize = T.Resize(img_size, interpolation=Image.BILINEAR)
        self.mask_resize = T.Resize(img_size, interpolation=Image.NEAREST)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

    def _process_mask(self, mask_pil: Image.Image):
        m = self.mask_resize(mask_pil)
        m = np.array(m, dtype=np.uint8)  # valores originales: {1=fondo, 2=borde, 3=mascota}
        if self.binary:
            m = ((m == 2) | (m == 3)).astype(np.float32)  # foreground=1
            return torch.from_numpy(m).unsqueeze(0)       # [1,H,W] para BCE/Dice binario
        else:
            # Remap a {0,1,2} -> {fondo,borde,mascota} para CrossEntropy
            remap = np.zeros_like(m, dtype=np.int64)
            remap[m == 1] = 0
            remap[m == 2] = 1
            remap[m == 3] = 2
            return torch.from_numpy(remap)                # [H,W] long

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img_pil, mask_pil = self.base[idx]
        img = self.img_resize(img_pil)
        mask_t = self._process_mask(mask_pil)

        # Augment simple: flip horizontal coherente
        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                if mask_t.ndim == 3:  # binario [1,H,W]
                    mask_t = TF.hflip(mask_t)
                else:                  # multiclase [H,W]
                    mask_t = torch.flip(mask_t, dims=[1])  # W dimension

        img = self.to_tensor(img)
        img = self.normalize(img)
        return img, mask_t

def show_images_and_masks(imgs, masks, mean=None, std=None, max_samples=6):
    """
    Visualiza pares (imagen, máscara) de un batch.

    imgs:   Tensor [B, 3, H, W]  (RGB) en [0,1] o normalizado
    masks:  Tensor [B, 1, H, W]  (binario o multiclase)
    mean, std: listas/tuplas para desnormalizar (ej. ImageNet)
    max_samples: máximo de ejemplos a mostrar
    """
    B = imgs.size(0)
    n = min(B, max_samples)

    plt.figure(figsize=(n*3, 6))

    for i in range(n):
        # --- Imagen ---
        img = imgs[i].detach().cpu().float()
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean).view(3, 1, 1)
            std_t = torch.tensor(std).view(3, 1, 1)
            img = img * std_t + mean_t
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()

        # --- Máscara ---
        mask = masks[i, 0].detach().cpu().numpy()

        # Imagen
        plt.subplot(2, n, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i+1}")

        # Máscara
        plt.subplot(2, n, n+i+1)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.title(f"Mask {i+1}")

    plt.tight_layout()
    plt.show()



def create_pets_loaders(root: str = "./data",
                        img_size: Tuple[int,int] = (256,256),
                        batch_size: int = 8,
                        num_workers: int = 2,
                        pin_memory: bool = True,
                        binary: bool = True,
                        augment: bool = True,
                        val_split: float = 0.15,
                        seed: int = 42) -> Dict[str, DataLoader]:
    
    """
    Crea DataLoaders para Oxford-IIIT Pet (segmentación).

    Args:
        batch_size: tamaño del batch (puedes seleccionarlo libremente)
        binary: True -> máscara binaria; False -> multiclase {0,1,2}
        augment: flips horizontales en train
        val_split: proporción de validación sacada del split 'trainval'
    """
    # Dataset base (trainval -> lo dividimos en train/val)
    full_train = OxfordPetsSeg(root=root, split="trainval",img_size=img_size, binary=binary,
                               augment=False, download=True)

    n = len(full_train)
    n_val = int(n * val_split)
    n_train = n - n_val

    # Reproducibilidad en el split
    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, lengths=[n_train, n_val], generator=g)

    # Activar augment solo en el subset de train (envolvemos con mismo dataset pero augment=True)
    # Usamos indices del subset para acceder al dataset base con augment deseado.

    class _SubsetWithAug(Dataset):
        def __init__(self, base_ds, indices, use_augment):
            self.base = base_ds
            self.indices = indices
            self.use_augment = use_augment

        def __len__(self): return len(self.indices)


        def __getitem__(self, i):
            idx = self.indices[i]
            # toggle augment dinámicamente
            prev = self.base.augment
            self.base.augment = self.use_augment
            out = self.base[idx]
            self.base.augment = prev
            return out

    train_ds = _SubsetWithAug(full_train, train_subset.indices, use_augment=augment)
    val_ds   = _SubsetWithAug(full_train, val_subset.indices,   use_augment=False)

    # Test oficial del dataset
    test_ds = OxfordPetsSeg(root=root, split="test",
                            img_size=img_size, binary=binary,
                            augment=False, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return {"train": train_loader, "val": val_loader, "test": test_loader}






