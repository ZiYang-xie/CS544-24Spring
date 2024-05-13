from .image_fitting import ImageFitting
from .image_classification import ImageClassification
from .continue_fitting import ContinueFitting
TASKS = {
    'image_fitting': ImageFitting,
    'image_classification': ImageClassification,
    'continue_fitting': ContinueFitting,
    'optimizer_choose': ImageFitting,
}