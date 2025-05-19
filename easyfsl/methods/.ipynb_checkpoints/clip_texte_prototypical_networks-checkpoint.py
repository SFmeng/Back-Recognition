from typing import Optional
import torch.nn.functional as F
import torch
from torch import Tensor, nn
from easyfsl.methods.utils import compute_prototypes

class TextPrototypicalNetworks(nn.Module):
    def __init__(
            self,
            backbone: Optional[nn.Module] = None,
            use_softmax: bool = False,
            feature_centering: Optional[Tensor] = None,
            feature_normalization: Optional[float] = None,
    ):

        super().__init__()

        self.backbone = backbone if backbone is not None else nn.Identity()
        self.use_softmax = use_softmax

        self.prototypes = torch.tensor(())  # 用来存储原型的
        self.support_features = torch.tensor(())
        self.support_labels = torch.tensor(())

        self.feature_centering = (
            feature_centering if feature_centering is not None else torch.tensor(0)
        )
        self.feature_normalization = feature_normalization


    def process_support_set(self, support_images: Tensor, support_labels: Tensor, ):
        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    def compute_features(self, images: Tensor) -> Tensor:

        original_features = self.backbone(images)
        centered_features = original_features - self.feature_centering
        if self.feature_normalization is not None:
            return nn.functional.normalize(centered_features, p=self.feature_normalization, dim=1)
        return centered_features

    def softmax_if_specified(self, output: Tensor, temperature: float = 1.0) -> Tensor:
        return (temperature * output).softmax(-1) if self.use_softmax else output

    def l2_distance_to_prototypes(self, samples: Tensor) -> Tensor:
        return -torch.cdist(samples, self.prototypes)

    def cosine_distance_to_prototypes(self, samples) -> Tensor:
        return (
                nn.functional.normalize(samples, dim=1)
                @ nn.functional.normalize(self.prototypes, dim=1).T
        )

    def compute_prototypes_and_store_support_set(self, support_images: Tensor, support_labels: Tensor, ):

        self.support_labels = support_labels  # 保存支持集标签
        self.support_features = self.compute_features(support_images)  # 利用主干提取特征
        self._raise_error_if_features_are_multi_dimensional(self.support_features)  # 确保特征是1维的
        self.prototypes = compute_prototypes(self.support_features, support_labels)  # 计算原型

    @staticmethod
    def _raise_error_if_features_are_multi_dimensional(features: Tensor):
        if len(features.shape) != 2:
            raise ValueError(
                "Illegal backbone or feature shape. "
                "Expected output for an image is a 1-dim tensor."
            )

    def forward(self, query_images: Tensor, ) -> Tensor:
        
        # print(query_images.shape,'========')
        query_features = self.compute_features(query_images)  # 25 = 5 way * 5query

        self._raise_error_if_features_are_multi_dimensional(query_features)

        # Compute the euclidean distance from queries to prototypes
        scores = self.l2_distance_to_prototypes(query_features)
        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False
