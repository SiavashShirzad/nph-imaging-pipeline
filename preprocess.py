import nibabel as nib
import numpy as np
import SimpleITK as sitk

from config import (N4_CONVERGENCE, N4_ITERATIONS, N4_SHRINK, REGISTRATION_SAMPLING,
                    REGISTRATION_SEED, REGISTRATION_SHRINK, REGISTRATION_SMOOTHING_MM)

# nibabel works in RAS, SimpleITK in LPS.
_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0])


def nib_to_sitk(img):
    affine = img.affine
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    image = sitk.GetImageFromArray(np.asarray(img.get_fdata(), dtype=np.float32).transpose(2, 1, 0))
    image.SetSpacing([float(s) for s in spacing])
    image.SetOrigin([float(o) for o in _RAS_TO_LPS @ affine[:3, 3]])
    image.SetDirection([float(d) for d in (_RAS_TO_LPS @ affine[:3, :3] / spacing).ravel()])
    return image


def sitk_to_nib(image):
    direction = np.array(image.GetDirection()).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = _RAS_TO_LPS @ direction @ np.diag(image.GetSpacing())
    affine[:3, 3] = _RAS_TO_LPS @ np.array(image.GetOrigin())
    data = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    return nib.as_closest_canonical(nib.Nifti1Image(data, affine))


def load_template(path=None):
    """MNI152 T1 template; nilearn's bundled ICBM 2009c 1 mm when no path is given."""
    if path is None:
        from nilearn.datasets import load_mni152_template
        img = load_mni152_template(resolution=1)
    else:
        img = nib.load(str(path))
    return nib_to_sitk(nib.as_closest_canonical(img))


def n4_bias_correction(image, shrink=N4_SHRINK, iterations=N4_ITERATIONS,
                       convergence=N4_CONVERGENCE):
    """N4 fitted on a shrunk copy, bias field applied at full resolution."""
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    mask = sitk.BinaryFillhole(sitk.BinaryMorphologicalClosing(mask, [2, 2, 2]))

    small = sitk.Shrink(image, [shrink] * 3)
    small_mask = sitk.Shrink(mask, [shrink] * 3)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(iterations))
    corrector.SetConvergenceThreshold(convergence)
    corrector.Execute(small, small_mask)

    log_bias = corrector.GetLogBiasFieldAsImage(image)
    return image / sitk.Exp(log_bias)


def rigid_register(fixed, moving):
    """Rigid (6 DOF) Mattes MI registration of `moving` onto `fixed`, coarse to fine."""
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                sitk.CenteredTransformInitializerFilter.MOMENTS)

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(REGISTRATION_SAMPLING, REGISTRATION_SEED)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4,
                                                    numberOfIterations=300,
                                                    relaxationFactor=0.5)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(list(REGISTRATION_SHRINK))
    method.SetSmoothingSigmasPerLevel(list(REGISTRATION_SMOOTHING_MM))
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)

    transform = method.Execute(fixed, moving)
    return transform, {
        'metric': float(method.GetMetricValue()),
        'iterations': int(method.GetOptimizerIteration()),
        'stop_condition': method.GetOptimizerStopConditionDescription(),
    }


def _euler_parameters(transform):
    """Rotation (deg) and translation (mm) of the composite rigid transform."""
    if isinstance(transform, sitk.CompositeTransform):
        transform = transform.GetBackTransform()
    euler = sitk.Euler3DTransform(transform)
    angles = np.degrees(euler.GetParameters()[:3])
    return [round(float(a), 3) for a in angles], [round(float(t), 3) for t in euler.GetTranslation()]


def preprocess(img, n4=True, register=True, template=None):
    """Optional N4 bias correction followed by rigid registration to MNI152.

    N4 runs in native space so the bias field is estimated on the original sampling; the
    corrected image is then registered and resampled once onto the template grid.
    Returns the preprocessed nibabel image, the native-to-template transform (or None)
    and a dict describing what was done.
    """
    image = nib_to_sitk(img)
    info = {'n4': bool(n4), 'mni_registration': bool(register)}

    if n4:
        image = n4_bias_correction(image)

    transform = None
    if register:
        if template is None:
            template = load_template()
        transform, stats = rigid_register(template, image)
        image = sitk.Resample(image, template, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        rotation, translation = _euler_parameters(transform)
        info.update(stats, rotation_deg=rotation, translation_mm=translation,
                    template_shape=list(template.GetSize()),
                    template_spacing_mm=[float(s) for s in template.GetSpacing()])

    return sitk_to_nib(image), transform, info
