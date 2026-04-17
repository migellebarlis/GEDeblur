# GEDeblur

A gray-box, explainable deep learning approach for blind deblurring of motion-blurred images.

## Overview

GEDeblur embeds a classical blind-deconvolution solver inside a learned multi-scale encoder-decoder so that the network is both accurate and interpretable. At each scale of the U-Net, a dedicated `Kernel` module performs an FFT-based alternating update that mirrors the classical variational formulation of blind deblurring:

1. A Wiener-like update that refines an intermediate sharp feature map from the current blur estimate and kernel.
2. A kernel update that refines the per-channel PSF from the current sharp estimate.

The sharp/kernel updates are differentiable and use a small set of learnable scalar parameters (`beta`, `gamma`, `zeta`, `lamda`) so training can tune the balance between data-fit and regularisation while the inner structure of the optimisation remains visible. Because the intermediate kernels, blur features, and sharp features at every scale are genuine tensors flowing through the forward pass, they can be inspected directly (see [explain.py](explain.py)) to explain what the model has learned.

Two families of models are provided: a convolutional U-Net variant (the main `convolution_7_32` model and a black-box ablation `convolution_7_32_full` without the `Kernel` modules) and a patch-based transformer variant (`transformer_*`) that performs per-patch Wiener deconvolution with a blur-similarity attention mechanism.

## Repository layout

```
.
├── train.py                    # Training loop for convolution_7_32 on GoPro
├── demo.py                     # Single-image inference on bundled sample/
├── explain.py                  # Dumps per-scale activations for interpretability
├── requirements.txt
├── model/
│   ├── convolution_7_32.py     # Main gray-box U-Net with Kernel modules
│   ├── convolution_7_32_full.py# Black-box U-Net ablation (no Kernel modules)
│   ├── transformer_1_4.py      # Transformer variants (patch-based deconvolution)
│   ├── transformer_2_4.py
│   ├── transformer_2_9.py
│   └── transformer_3_4.py
├── dataset/
│   ├── dataloader.py           # Unified DeblurringDataLoader dispatcher
│   ├── gopro.py                # GoPro (GOPRO_Large)
│   ├── hide.py                 # HIDE
│   ├── realblurr.py            # RealBlur-R
│   ├── realblurj.py            # RealBlur-J
│   └── div2k.py                # DIV2K
├── test/
│   ├── gopro.py                # PSNR/ISNR/SSIM benchmark on GoPro
│   ├── hide.py
│   ├── realblurr.py            # ECC-aligned masked PSNR/SSIM
│   ├── realblurj.py            # ECC-aligned masked PSNR/SSIM
│   └── div2k.py
├── utils/
│   └── utils_deblur.py         # psf2otf / otf2psf / DCT helpers
├── checkpoint/                 # Pre-trained weights (.pth)
└── sample/                     # Example blurry/sharp/restored triplet
```

## Model variants

| Module | Checkpoint | Description |
| --- | --- | --- |
| `model.convolution_7_32.Deblur` | `checkpoint/convolution_7_32.pth` | Main gray-box model. Three encoder stages plus bottleneck plus three decoder stages, each with a `Kernel` block performing FFT-domain Wiener/kernel updates over separate `kern`, `blur`, and `sharp` streams. `in_channels=3`, `embed_channels=32`. |
| `model.convolution_7_32_full.Deblur` | `checkpoint/convolution_7_32_full.pth` | Black-box ablation with the same U-Net topology but no `Kernel` modules; useful for comparing the contribution of the explainable update blocks. |
| `model.transformer_1_4.Xformer` | `checkpoint/transformer_1_4.pth` | Patch-based transformer variant: `BlurSimilarity` attention, `SharpBlurFeature` extractor, and DCT-domain `KernelEstimate` / `ImageEstimate` blocks. |
| `model.transformer_2_4.Xformer2` | `checkpoint/transformer_2_4.pth` | Two-stage transformer. |
| `model.transformer_2_9.Xformer2` | `checkpoint/transformer_2_9.pth` | Two-stage transformer with nine attention heads. |
| `model.transformer_3_4.Xformer3` | `checkpoint/transformer_3_4.pth` | Three-stage transformer. |

## Installation

The project targets Python 3.10+ and PyTorch 2.1.2 with CUDA 11.8/12.1. A compatible GPU build of PyTorch should be installed before the rest of the dependencies; follow the instructions at [pytorch.org](https://pytorch.org/get-started/previous-versions/) for your platform, then install the remaining packages:

```bash
pip install -r requirements.txt
```

Pinned versions are listed in [requirements.txt](requirements.txt) (`torch==2.1.2`, `torchvision==0.16.2`, `einops==0.7.0`, `timm==0.9.12`, `opencv-python==4.8.1.78`, `scikit-image==0.22.0`, `scipy==1.11.4`, `numpy==1.26.2`, `tqdm==4.66.1`).

## Datasets

The dataset classes in [dataset/](dataset/) expect the standard "blur / sharp" pair layout. For GoPro (see [dataset/gopro.py](dataset/gopro.py)):

```
<dataset_root>/
├── train/
│   └── <scene>/
│       ├── blur/*.png
│       └── sharp/*.png
└── test/
    └── <scene>/
        ├── blur/*.png
        └── sharp/*.png
```

Dataset roots are currently hardcoded in two places that you need to edit for your environment:

- [dataset/dataloader.py](dataset/dataloader.py) - used by [train.py](train.py). Defaults: `G:/GOPRO_Large`, `G:/RealBlur`, `G:/HIDE`.
- Each script under [test/](test/) - defaults: `dir_dataset = 'G:/GOPRO_Large'` (and analogous entries for the other datasets).

Supported datasets:

- **GoPro** (`GOPRO_Large`) - primary training/evaluation set.
- **HIDE** - cross-dataset evaluation.
- **RealBlur-R** and **RealBlur-J** - real-world evaluation; the test scripts perform ECC image alignment and masked PSNR/SSIM following the RealBlur protocol.
- **DIV2K** - used for additional evaluation.

## Quick demo

A small blurry/sharp pair is bundled under [sample/](sample/). To run inference with the pretrained gray-box model on a CPU:

```bash
python demo.py
```

[demo.py](demo.py) loads `checkpoint/convolution_7_32.pth`, runs `Deblur(in_channels=3, embed_channels=32)` on `sample/y.png`, saves the restored image to `sample/xhat.png`, displays it with matplotlib, and prints a line such as:

```
PSNR: 30.412, ISNR: 7.894
```

## Training

To train the main gray-box model on GoPro:

```bash
python train.py
```

Defaults used by [train.py](train.py):

- Model: `Deblur(in_channels=3, embed_channels=32)`.
- Dataset: `DeblurringDataLoader(set_name='gopro', batch_size=20, image_size=256, num_workers=2)`.
- Optimiser: `AdamW(lr=3e-4, weight_decay=1e-4)`.
- Scheduler: `CosineAnnealingLR(T_max=2000, eta_min=1e-6)`.
- Loss: `MSELoss` for both training and validation.
- Epochs: `2000`, random seed `1234`.
- Checkpoint path: `./checkpoint/convolution_7_32.pth`. If the file already exists it is reloaded (model, optimiser, scheduler, best validation loss, last epoch) so training resumes automatically.

Progress bars print running `loss`, `psnr`, and `isnr` for both training and validation phases.

## Evaluation

Per-dataset benchmark scripts live under [test/](test/). They load `checkpoint/convolution_7_32.pth`, iterate the dataset's `test` split, pad each image to a square multiple of 128 pixels before feeding it through the model, crop back to the original size, and report PSNR, ISNR, and SSIM.

Example (run from the repository root so the relative imports resolve):

```bash
python -m test.gopro
python -m test.hide
python -m test.realblurj
python -m test.realblurr
python -m test.div2k
```

Each run writes:

- Restored PNGs under `./result/convolution_7_32/<dataset>/<scene>/blur/...`.
- Per-image metrics to `./result/convolution_7_32/<dataset>/psnr_isnr_ssim.txt` and `.csv`.
- A final aggregated line with the mean PSNR/ISNR/SSIM.

RealBlur-R and RealBlur-J additionally perform ECC (homography) alignment between the restored and ground-truth images, and compute masked PSNR/SSIM following the official RealBlur evaluation protocol.

## Explainability

[explain.py](explain.py) exercises the gray-box model on a single image and dumps the intermediate tensors that justify the "explainable" label. It registers forward hooks on:

- `input_proj`, `output_proj`
- `encode_0`, `encode_1`, `encode_2`, `bottleneck` (per-scale blurry features)
- `kernel_0`, `kernel_1`, `kernel_2`, `kernel_bottleneck` (estimated sharp features)
- `downsample_00`, `downsample_10`, `downsample_20` (per-scale kernel tensors)
- `decode_0`, `decode_1`, `decode_2` (per-scale reconstructed sharp features)

For each scale it saves a grid of PNGs into `./activations/`, covering the input projection weights, the learned blur feature maps, the estimated per-channel PSFs (plus their magnitude spectra via `psf2otf`), the intermediate sharp images, and the decoder outputs.

To run it:

```bash
mkdir activations
# place an input image named 09.png in the repository root
python explain.py
```

## Pre-trained checkpoints

The [checkpoint/](checkpoint/) directory ships with all weights referenced above:

- `convolution_7_32.pth` - loaded by `model.convolution_7_32.Deblur` and used by [demo.py](demo.py), [train.py](train.py), [explain.py](explain.py), and every script under [test/](test/).
- `convolution_7_32_full.pth` - loaded by `model.convolution_7_32_full.Deblur` (black-box ablation).
- `transformer_1_4.pth`, `transformer_2_4.pth`, `transformer_2_9.pth`, `transformer_3_4.pth` - loaded by the matching `model.transformer_*` modules.

Each checkpoint stores `model_state`, `optim_state`, `sched_state`, `accuracy` (best validation loss), and `epoch`.

## Notes and known caveats

- [dataset/dataloader.py](dataset/dataloader.py) imports the dataset modules as `from gopro import GoProDataset`, `from hide import HIDEDataset`, etc., rather than as relative imports. It therefore needs to be run with [dataset/](dataset/) on `sys.path` (running `python train.py` from the repository root works because `train.py` imports `dataset.dataloader` which transitively picks up the bundled dataset files only if the imports are patched, or if you add the `dataset/` directory to `PYTHONPATH`).
- [explain.py](explain.py) expects a writable `./activations/` directory and an input image named `09.png` in the working directory; create both before running.
- All dataset paths are hardcoded (defaults point to drive `G:`); update them in [dataset/dataloader.py](dataset/dataloader.py) and the corresponding [test/](test/) scripts before training or evaluating.
