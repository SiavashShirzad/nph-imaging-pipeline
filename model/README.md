# Models

The trained models are not stored in git. Download them with

```bash
python download_models.py
```

which fetches `nph-models-v1.0.zip` from the repository's
[v1.0 release](https://github.com/SiavashShirzad/nph-imaging-pipeline/releases/tag/v1.0), checks
its SHA-256 and unpacks it here. To install by hand, download the zip from the release page and
unzip it in the repository folder so that this layout results:

```
model/
    acpc/AC_PC_IVF_BestModel.pth        BrainSignsNET AC / PC / IVF landmark network
    acpc/AC_PC_IVF_BestModel.json       landmark order and training metadata
    icv/                                nnU-Net intracranial volume model
        dataset.json
        plans.json
        fold_all/checkpoint_final.pth
    ventricles/                         nnU-Net lateral ventricle model
        dataset.json
        plans.json
        fold_0/checkpoint_best.pth
    ca/best_pretrained_segmentation_model_2.pth   callosal angle ventricle segmentation
    desi/best_model.pth                 Sylvian / high-convexity segmentation (DESI)
```

The models are covered by the same non-commercial licence as the code (see `LICENSE`).
