import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from model.backbone.ResNet import Res_DeepLabV3P
from model.backbone.SegFormer import SegFormer
from model.backbone.DeepLab import DeepLabHeadV3Plus
from model.UperNet import UPerNet
from model.SAA import SAA
from model.CCA import CCA


class Sem_Enc(nn.Module):
    """
    Semantic Encoding Module.

    This module converts semantic features from the ResNet backbone into encodings.
    It processes the high-level semantic features using a DeepLabV3+ head followed by
    a series of depthwise convolutions and pooling operations to generate encodings
    of shape (B, num_classes, 1).

    Args:
        num_classes (int): Number of target classes for segmentation.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Initializes the Sem_Enc module.

        Args:
            num_classes (int): Number of target classes for segmentation.
        """
        super(Sem_Enc, self).__init__()

        # Initialize DeepLabV3+ head for projecting semantic features
        self.projection: DeepLabHeadV3Plus = DeepLabHeadV3Plus(num_classes=num_classes)

        # Depthwise convolution layers with specified kernel sizes and pooling
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=7,
            stride=1,
            padding=4,
            groups=num_classes,  # Depthwise convolution
            bias=False,
        )
        self.pool1: nn.AvgPool2d = nn.AvgPool2d(kernel_size=6)

        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=num_classes,  # Depthwise convolution
            bias=False,
        )
        self.pool2: nn.AvgPool2d = nn.AvgPool2d(
            kernel_size=4
        )  # Reduces spatial dimensions from 16 to 4

        self.conv3: nn.Conv2d = nn.Conv2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=4,
            stride=1,
            padding=0,
            groups=num_classes,  # Depthwise convolution
            bias=False,
        )  # Reduces spatial dimensions from 4 to 1

        # Batch normalization and ReLU activation
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(num_classes)
        self.relu: nn.ReLU = nn.ReLU()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the Sem_Enc module.

        Args:
            features (List[torch.Tensor]): List of feature maps from the ResNet backbone.
                                           Expected to contain [layer1, layer2, layer3, layer4].

        Returns:
            torch.Tensor: Semantic encodings with shape (B, num_classes).
        """
        # Apply DeepLabV3+ projection to semantic features
        # Input: {'low_level': features[0], 'out': features[3]}
        x: torch.Tensor = self.projection(
            {"low_level": features[0], "out": features[3]}
        )  # Shape: (B, num_classes, H, W)

        # First convolution and pooling layer
        conv: torch.Tensor = self.conv1(x)  # Shape: (B, num_classes, H, W)
        conv = self.pool1(conv)  # Shape: (B, num_classes, H//6, W//6)
        conv = self.bn(conv)  # Shape: (B, num_classes, H//6, W//6)
        conv = self.relu(conv)  # Shape: (B, num_classes, H//6, W//6)

        # Second convolution and pooling layer
        conv = self.conv2(conv)  # Shape: (B, num_classes, H//6, W//6)
        conv = self.pool2(conv)  # Shape: (B, num_classes, H//24, W//24)
        conv = self.bn(conv)  # Shape: (B, num_classes, H//24, W//24)
        conv = self.relu(conv)  # Shape: (B, num_classes, H//24, W//24)

        # Third convolution layer to reduce spatial dimensions to 1x1
        conv = self.conv3(conv)  # Shape: (B, num_classes, 1, 1)
        conv = self.bn(conv)  # Shape: (B, num_classes, 1, 1)
        conv = self.relu(conv)  # Shape: (B, num_classes, 1, 1)

        # Squeeze the last two dimensions to obtain (B, num_classes)
        return conv.squeeze(3)  # Shape: (B, num_classes, 1) -> (B, num_classes)


class GlassSemNet(nn.Module):
    """
    GlassSemNet Model for Semantic Segmentation.

    This model integrates spatial and semantic backbones, semantic encodings,
    Scene Aware Activation (SAA) modules, Context Correlation Attention (CCA) module,
    and a decoder (UPerNet) to produce segmentation outputs.

    Architecture Components:
        - Spatial Backbone: SegFormer
        - Semantic Backbone: Res_DeepLabV3P
        - Semantic Encoding: Sem_Enc
        - SAA Modules: SAA0, SAA1, SAA2
        - CCA Module: CCA3
        - Decoder: UPerNet
        - Auxiliary Outputs: aux1, aux2

    Args:
        None
    """

    def __init__(self) -> None:
        """
        Initializes the GlassSemNet model.
        """
        super(GlassSemNet, self).__init__()

        # Define the number of segmentation classes
        self.num_classes: int = 43

        # Initialize the spatial backbone (SegFormer)
        self.spatial_backbone: SegFormer = SegFormer()

        # Initialize the semantic backbone (ResNet with DeepLabV3+)
        self.semantic_backbone: Res_DeepLabV3P = Res_DeepLabV3P()

        # Initialize the semantic encoding module
        self.sem_enc: Sem_Enc = Sem_Enc(num_classes=self.num_classes)

        # Initialize Scene Aware Activation (SAA) modules for different feature levels
        self.saa0: SAA = SAA(
            spatial_dim=64, semantic_dim=256, semantic_assert=self.num_classes
        )
        self.saa1: SAA = SAA(
            spatial_dim=128, semantic_dim=512, semantic_assert=self.num_classes
        )
        self.saa2: SAA = SAA(
            spatial_dim=320, semantic_dim=1024, semantic_assert=self.num_classes
        )

        # Initialize Context Correlation Attention (CCA) module for the highest feature level
        self.cca3: CCA = CCA(
            spatial_dim=512,
            semantic_dim=2048,
            transform_dim=1024,
            semantic_assert=self.num_classes,
        )

        # Initialize auxiliary convolution layers for intermediate outputs (optional)
        self.aux1: nn.Conv2d = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
        self.aux2: nn.Conv2d = nn.Conv2d(
            in_channels=1024, out_channels=1, kernel_size=1
        )

        # Initialize the decoder (UPerNet) for final segmentation output
        self.decoder: UPerNet = UPerNet()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GlassSemNet model.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 3, H, W).

        Returns:
            torch.Tensor: Segmentation output tensor with shape (B, num_class, H', W'),
                          where (H', W') is determined by the decoder's output resolution.
        """
        # ---------------------------
        # Spatial Backbone Forward Pass
        # ---------------------------
        spatial_feats: List[torch.Tensor] = self.spatial_backbone(x)
        # spatial_feats: List of feature maps from SegFormer, e.g., [P2, P3, P4, P5]

        # ---------------------------
        # Semantic Backbone Forward Pass
        # ---------------------------
        resnet_out: Dict[str, List[torch.Tensor]] = self.semantic_backbone(x)
        semantic_feats: List[torch.Tensor] = resnet_out["backbone"]
        semantic_lowlevel: torch.Tensor = resnet_out["layer0"]
        # semantic_feats: List of semantic feature maps from ResNet, e.g., [C1, C2, C3, C4]

        # ---------------------------
        # Semantic Encodings
        # ---------------------------
        sem_enc: torch.Tensor = self.sem_enc(semantic_feats)
        # sem_enc: Semantic encodings with shape (B, num_classes)

        # ---------------------------
        # Scene Aware Activation (SAA) Modules
        # ---------------------------
        saa0: torch.Tensor = self.saa0(
            spatial_feature=spatial_feats[0],
            semantic_feature=semantic_feats[0],
            sem_encod=sem_enc,
        )
        saa1: torch.Tensor = self.saa1(
            spatial_feature=spatial_feats[1],
            semantic_feature=semantic_feats[1],
            sem_encod=sem_enc,
        )
        saa2: torch.Tensor = self.saa2(
            spatial_feature=spatial_feats[2],
            semantic_feature=semantic_feats[2],
            sem_encod=sem_enc,
        )

        # ---------------------------
        # Context Correlation Attention (CCA) Module
        # ---------------------------
        cca3: torch.Tensor = self.cca3(
            spatial_feature=spatial_feats[3],
            semantic_feature=semantic_feats[3],
            sem_encod=sem_enc,
        )

        # ---------------------------
        # Decoder Preparation
        # ---------------------------
        # Concatenate spatial, semantic, and activated features for each level
        l0: torch.Tensor = torch.cat(
            [spatial_feats[0], semantic_feats[0], saa0], dim=1
        )  # Shape: (B, C0 + C0 + C0, H0, W0)
        l1: torch.Tensor = torch.cat(
            [spatial_feats[1], semantic_feats[1], saa1], dim=1
        )  # Shape: (B, C1 + C1 + C1, H1, W1)
        l2: torch.Tensor = torch.cat(
            [spatial_feats[2], semantic_feats[2], saa2], dim=1
        )  # Shape: (B, C2 + C2 + C2, H2, W2)
        l3: torch.Tensor = torch.cat(
            [spatial_feats[3], semantic_feats[3], cca3], dim=1
        )  # Shape: (B, C3 + C3 + C3, H3, W3)

        # Prepare the list of feature maps for the decoder
        # Typically, decoder expects [low_level, P2, P3, P4, P5]
        decoder_feats: List[torch.Tensor] = [semantic_lowlevel, l0, l1, l2, l3]

        # ---------------------------
        # Decoder Forward Pass
        # ---------------------------
        out: torch.Tensor = self.decoder(decoder_feats)
        # out: Segmentation output from UPerNet, shape depends on UPerNet configuration

        # Optional: Auxiliary outputs (commented out)
        # aux_out1: torch.Tensor = self.aux1(saa1)  # Shape: (B, 1, H1, W1)
        # aux_out2: torch.Tensor = self.aux2(cca3)  # Shape: (B, 1, H3, W3)
        # out = out + aux_out1 + aux_out2  # Combine main and auxiliary outputs

        return out  # Final segmentation output


# Example usage:
# if __name__ == '__main__':
#     # Create a random input tensor with batch size 2, 3 channels, and 384x384 spatial dimensions
#     x: torch.Tensor = torch.rand(2, 3, 384, 384)

#     # Initialize the GlassSemNet model
#     model: GlassSemNet = GlassSemNet()

#     # Perform a forward pass
#     out: torch.Tensor = model(x)

#     # Print the output shape
#     print(out.shape)  # Expected shape: (2, num_class, H', W')
