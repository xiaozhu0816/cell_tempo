from .resnet import ResNetClassifier, build_model
from .regression_resnet import ResNetRegressor, build_regression_model
from .multitask_resnet import MultiTaskResNet, build_multitask_model

__all__ = [
	"ResNetClassifier",
	"build_model",
	"ResNetRegressor",
	"build_regression_model",
	"MultiTaskResNet",
	"build_multitask_model",
]
