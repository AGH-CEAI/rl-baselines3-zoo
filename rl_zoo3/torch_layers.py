from typing import Type
from collections import OrderedDict
from abc import ABC, abstractmethod
from numpy import ceil

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space

class AbstractVisionExtractor(BaseFeaturesExtractor, ABC):
    """
    Load pre-trained model from torchvision library as feature extractor.
    List of avaiable models: https://pytorch.org/vision/main/models.html

    :param observation_space:
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8)
        and bounds (values in [0, 255]).
    :param model_name: the name of the model in the PascalCase format.
    :param weights_id: the name of the trained weights (torchvision API).
    :param cut_on_layer: the name of the layer to cut the head from the backbone.
    :param add_linear_layer: enables creation of linear layer
        for enforcing output dimension.
    :param linear_features_dim: Number of features extracted.
            This corresponds to the number of unit for the last layer.
    :param linear_activation_fn: The activation function to use after each layer.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        normalized_image: bool,
        linear_features_dim: int,
        linear_activation_fn: Type[nn.Module],
        stacking_frames: int | None = None,
        frame_channels: int = 3,
    ):
        super().__init__(observation_space, linear_features_dim)
        assert isinstance(observation_space, spaces.Box), (
            "PreTrainedVisionExtractor must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        is_single_frame = stacking_frames is None or stacking_frames < 2
        if is_single_frame:
            assert is_image_space(observation_space, check_channels=True, normalized_image=normalized_image), (
                f"You should use a VisionExtractor only with images not with {observation_space}.\n"
                "If the `stackig_frames` is greater than 1, please set the Extractor params `stackig_frames`and `frame_channels` accordingly."
            )
        else:
            n_channels = observation_space.shape[0]
            modulo = n_channels % frame_channels
            divison = ceil(n_channels / frame_channels)        
            assert modulo == 0 and  divison == stacking_frames, (
                f"The number of passed channels ({n_channels}) is not matching stacked frames ({stacking_frames}) and "
                f"defined frame channels ({frame_channels}). Check `stacking_frames` and `frame_channels` configuration for given observation."
            )
            
        feature_extractor = self._prepare_feature_extractor()
        flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with th.no_grad():
            obs = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = flatten(feature_extractor(obs)).squeeze().shape[0]
            
        linear = nn.Sequential(nn.Linear(n_flatten, linear_features_dim), linear_activation_fn())
        self.fe_model = nn.Sequential(feature_extractor, flatten, linear)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.fe_model(observations)

    @abstractmethod
    def _prepare_feature_extractor(self):
        raise NotImplementedError()

class CustomVisionExtractor(AbstractVisionExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        custom_feature_extractor: Type[nn.Module],
        normalized_image: bool = False,
        linear_features_dim: int = 512,
        linear_activation_fn: Type[nn.Module] = nn.ReLU,
        stacking_frames: int = None,
        frame_channels: int = 3,
    ):
        self._custom_model = custom_feature_extractor
        super().__init__(
            observation_space,
            normalized_image,
            linear_features_dim,
            linear_activation_fn,
            stacking_frames,
            frame_channels,
        )

    def _prepare_feature_extractor(self):
        return self._custom_model
    
class PreTrainedVisionExtractor(AbstractVisionExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        model_name: str = None,
        weights_id: str | None = None,
        cut_on_layer: str = None,
        normalized_image: bool = False,
        linear_features_dim: int = 512,
        linear_activation_fn: Type[nn.Module] = nn.ReLU,
        stacking_frames: int = None,
        frame_channels: int = 3,
    ):
        self._import_torchvision()
        self._model_name = model_name
        self._weights_id = weights_id
        self._cut_on_layer = cut_on_layer
        super().__init__(
            observation_space,
            normalized_image,
            linear_features_dim,
            linear_activation_fn,
            stacking_frames,
            frame_channels,
        )

    def _import_torchvision(self):
        try:
            self._thvision = __import__("torchvision")
        except ImportError:
            raise ImportError(
                "Can't use PreTrainedVisionExtractor without torchvision. Please install it (`pip install torchvision`)."
            )

    def _prepare_feature_extractor(self):
        pretrained_model = self._load_vision_model(self._model_name, self._weights_id)
        return self._cut_head_layers(pretrained_model, self._cut_on_layer)

    def _load_vision_model(self, model_name: str, weights_id: str | None = None) -> nn.Module:
        try:
            weights = weights_id if weights_id is None else self._thvision.models.get_weight(weights_id)
            model = self._thvision.models.get_model(model_name, weights=weights)

            # TODO add feature to unfreeze speecific layers
            for param in model.parameters():
                param.requires_grad = False

            return model
        except ValueError as e:
            raise ValueError(
                f"{e}.\nFailed to load the '{model_name}' model with '{weights_id}' weights. Ensure that the name is in "
                f"the PascalCase format and it is listed in https://pytorch.org/vision/main/models.html."
            )

    def _cut_head_layers(self, model: nn.Module, cut_layer: str) -> nn.Module:
        layers = OrderedDict()

        for layer_name, layer in model.named_children():
            if layer_name == cut_layer:
                break
            layers[layer_name] = layer

        return nn.Sequential(layers)