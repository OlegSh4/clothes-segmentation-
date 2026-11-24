# Clothes Segmentation: U-Net vs SegFormer

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-SegFormer-FFD21E?logo=huggingface&logoColor=black)

Semantic segmentation of clothing items on photos of people (59 classes, including background). I implemented U-Net from scratch and compared it with a fine-tuned SegFormer transformer.

Coursework, Financial University (Moscow).

## Results

| Model | Training | Validation pixel accuracy |
|---|---|---|
| **SegFormer-B0** (fine-tuned from `nvidia/segformer-b0-finetuned-ade-512-512`) | 10 epochs | **88.9%** |
| U-Net (from scratch) | 150 epochs | reported per class (IoU) in the notebook |

The pretrained SegFormer reached high pixel accuracy after only a few epochs, which shows the value of transfer learning on a small dataset.

## Approach

1. **Dataset:** the People Clothing Segmentation dataset. I wrote a custom `ClothesSegmentationDataset` that returns the image and an integer mask, with shared transforms for both.
2. **U-Net:** implemented the encoder–decoder architecture with skip connections (`ConvTranspose2d` upsampling). The output is an `n_classes × H × W` tensor, trained with cross-entropy and tracked with per-class IoU.
3. **SegFormer:** loaded a pretrained SegFormer-B0 from Hugging Face, replaced the classification head for 59 classes and fine-tuned it.
4. **Visualization:** compared ground-truth masks with model predictions.

## Tech stack

Python · PyTorch · torchvision · Hugging Face Transformers · Matplotlib

## How to run

```bash
pip install torch torchvision transformers matplotlib pillow tqdm
```

Place `people-clothing-segmentation.zip` next to the notebook and run `Сегментация.ipynb`. A GPU is recommended.
