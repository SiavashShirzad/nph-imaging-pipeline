import numpy as np
from scipy.ndimage import label


def build_predictor(model_folder, folds, checkpoint, device):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=device.type == 'cuda',
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=tuple(folds),
        checkpoint_name=checkpoint
    )
    predictor.image_reader_writer_class = SimpleITKIO
    return predictor


def predict(predictor, img):
    data = np.expand_dims(img.get_fdata(), axis=0)
    properties = {
        'spacing': img.header.get_zooms()[:3],
        'shape': img.shape,
        'affine': img.affine,
    }
    seg = predictor.predict_single_npy_array(data, properties, None, None, False)
    return seg.astype(np.uint8)


def largest_component(mask):
    labelled, n = label(mask > 0, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(labelled.ravel())
    return (labelled == sizes[1:].argmax() + 1).astype(mask.dtype)
