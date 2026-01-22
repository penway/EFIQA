<img width="2400" height="400" alt="AAA" src="https://github.com/user-attachments/assets/93157862-5734-43f5-b923-29faf69f55c1" />

# EFIQA: Explainable Fundus Image Quality Assessment via Anatomical Priors
We ground fundus quality assessment in anatomical integrity rather than subjective labels, enabling an unsupervised approach with spatial explainability and robust cross-dataset generalization.

Currently under review in MIDL2026: https://openreview.net/forum?id=b9TBF3O88T

## Inference
Test our model directly online at: https://huggingface.co/spaces/penway47/EFIQA. You can also download script and weight there.

## Training
### Preparation
Create the vitual environment you like, for example:
```bash
conda create -n EFIQA python=3.13
```
Then install this repo:
```bash
conda activate EFIQA
cd EFIQA
pip install -e .
```

### Training VUAD
#### Preparing the dataset
You can use any dataset and segmentor you like, and could be binary segmentation or artery/vien segmentaion. To reproduce the results, we used the [Messidor-2](https://www.adcis.net/en/third-party/messidor2/) dataset. Cropping the edge to square and use [RRW-Net](https://github.com/j-morano/rrwnet) for segmentation, you can follow the instruction in [R2-V2](https://github.com/j-morano/R2-V2) and we used the bv model. We filter out several images as is shown in `src/vuad/config/Messidor_NSFS.txt`. For easy retraining the model, we will upload the segmented data.

#### Running the training
After getting the data, you can run the training with
```bash
python -m vuad.train --config-name=vuad_default data.path=/path/to/dataset
```
or you can also directly change the dataset path and output path in `src/vuad/config/vuad_default.yaml`.

#### Inference VUAD and prepare for adapter data
To prepare for training the adapter, you have to segment a dataset with degradation. Although we trained with an internal dataset, you can get almost the same performance from [MSHF dataset](https://figshare.com/articles/figure/MSHF_A_Multi-Source_Heterogeneous_Fundus_MSHF_Dataset_for_Image_Quality_Assessment/21507564?file=39485878).

```bash
python -m vuad.infer \
    -i .../seg/bv \
    -o .../seg/vuad \
    -m .../model_epoch_100.pth \
    --model_config=src/vuad/config/vuad_default.yaml
```

### Training adapter
The adapter is called `badish` in this repo.

#### Extract DINO feature
You should apply access for DINOv3 weights in huggingface, then extract features:
```bash
python -m badish.scripts.extract_dino \
        -i .../pixels \
        -o .../dino_emb \
        --size 224 --fp32 \
        --model_id facebook/dinov3-vitl16-pretrain-lvd1689m \
        --revision ea8dc2863c51be0a264bab82070e3e8836b02d51
```

Then compile dataset together using the following command, you have to change input and output paths in the `src/badish/scripts/prepare_patch_dataset.py`.
```
python -m badish.scripts.prepare_patch_dataset
```

Then train the model with this, rememeber to change the dataset path in `src/badish/config/badish_default.yaml`.
```bash
python -m badish.train --config-name=badish_default
```

Finally you can infer a whole dataset and get raw patch-level score with
```bash
python -m badish.infer \
  -i .../MSHF/emb_dino \
  -o .../badish_dino \
  -m .../model_epoch20.pth \
  -c src/badish/config/badish_default.yaml
```
