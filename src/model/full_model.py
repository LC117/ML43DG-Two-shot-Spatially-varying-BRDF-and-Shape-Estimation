
from src.model.shape_model import ShapeNetwork
from src.model.illumination_model import IlluminationNetwork
from src.model.svbrdf_model import SVBRDF_Network
from src.model.joint_model import JointNetwork

from utils.preprocessing_utils import read_image, read_mask


def full_inference(path_to_default_img, path_to_flash, path_to_mask):
    
    # Default parameters for the inference:
    parameters = [
        ParameterNames.INPUT_1_LDR if isLdr else ParameterNames.INPUT_1,
        ParameterNames.INPUT_2_LDR if isLdr else ParameterNames.INPUT_2,
        ParameterNames.MASK,
    ]
    default_image =  read_image(path_to_default_img)
    flash_image =  read_image(path_to_flash)
    mask = read_mask(path_to_mask)