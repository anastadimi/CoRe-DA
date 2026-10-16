from typing import Optional

import torch
import torch.nn as nn

from i3d import InceptionI3d


class I3DEncoder(nn.Module):
    """
    I3D-based feature extractor that processes videos in temporal snippets, and
    extracts spatiotemporal features using I3D.

    Parameters
    ----------
    feature_dim : int, optional
        The output dimension of the intermediate feature vector before regression (default: 256).
    snippet_size : int, optional
        Number of consecutive frames per snippet (default: 12).
    pretrained : bool, optional
        Whether to load pretrained weights for I3D base model (default: True). It loads the Kinetics-400 weights.
    weight_path : str, optional
        Path to the pretrained weight file (default: 'rgb_imagenet.pt'). Don't be fooled by the name, it's Kinetics-400.
    freeze_until : Optional[str], optional
        Freeze all layers up to but NOT including this endpoint name (default: None (not freezing)).
        Must be one of InceptionI3d.VALID_ENDPOINTS. Use None to disable freezing.
    freeze_bn_stats : bool, optional
        If True, BatchNorm3d layers in the frozen part are set to eval() to stop running stats updates (default: True).
    dropout_keep_prob: float
        Dropout probability for the base InceptionI3d model (default: 0.5).
    """

    def __init__(
        self,
        feature_dim: int = 256,
        snippet_size: int = 12,
        pretrained: bool = True,
        weight_path: str = "rgb_imagenet.pt",
        freeze_until: Optional[str] = None,  # default: no freezing
        freeze_bn_stats: bool = True,
        dropout_keep_prob: float = 0.5,
    ):
        super().__init__()

        self.snippet_size = snippet_size

        # Initialize I3D
        self.base = InceptionI3d(dropout_keep_prob=dropout_keep_prob)

        if pretrained:
            self._load_pretrained_weights(weight_path)

        # Optionally freeze backbone up to (but not including) a target endpoint
        if freeze_until is not None:
            self._freeze_backbone_until(freeze_until, freeze_bn_stats=freeze_bn_stats)

        # Replace logits layer with new output size
        self.base.replace_logits(num_classes=feature_dim)

    def _load_pretrained_weights(self, weight_path: str):
        try:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.base.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from '{weight_path}'")
        except FileNotFoundError:
            raise FileNotFoundError(f"Pretrained weights not found at: {weight_path}")
        except RuntimeError as e:
            raise RuntimeError(f"Error loading pretrained weights: {e}")

    def _freeze_backbone_until(self, endpoint: str, freeze_bn_stats: bool = True):
        """
        Freeze all layers up to but NOT including the given endpoint.

        Parameters
        ----------
        endpoint : str
            Target endpoint name (must exist in InceptionI3d.VALID_ENDPOINTS).
        freeze_bn_stats : bool
            If True, sets BatchNorm3d modules within the frozen part to eval().
        """
        if endpoint not in self.base.VALID_ENDPOINTS:
            raise ValueError(
                f"'freeze_until' must be one of {self.base.VALID_ENDPOINTS}, got {endpoint}."
            )

        # Freeze endpoints strictly before the target
        target_idx = self.base.VALID_ENDPOINTS.index(endpoint)

        for ep_name in self.base.VALID_ENDPOINTS[:target_idx]:
            if ep_name in self.base.end_points:
                module = self.base._modules[ep_name]  # same modules that forward() uses
                for p in module.parameters():
                    p.requires_grad = False
                if freeze_bn_stats:
                    for m in module.modules():
                        if isinstance(m, nn.BatchNorm3d):
                            # stop updating running stats; also makes affine params not train if requires_grad=False above
                            m.eval()

        # Note: we intentionally do NOT touch modules at and after `endpoint` (e.g., Mixed_5c, avg_pool, logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, T, H, W) where T % snippet_size == 0
        returns: (B, num_snippets, feature_dim)
        """
        B, C, T, H, W = x.shape
        assert T % self.snippet_size == 0, "T must be divisible by snippet_size"
        num_snippets = T // self.snippet_size

        # Split into snippets
        x = x.reshape(B * num_snippets, C, self.snippet_size, H, W)

        # Extract features
        x = self.base.extract_features(x)  # (B*num_snippets, 1024, t, 1, 1)
        x = self.base.logits(x)  # (B*num_snippets, feature_dim, t, 1, 1)
        x = x.squeeze(-1).squeeze(-1).mean(2)  # (B*num_snippets, feature_dim)
        f = x.view(B, num_snippets, -1)  # (B, num_snippets, feature_dim)

        return f


class CoRe_DA(nn.Module):
    def __init__(
        self,
        feat_args: dict = None,
        rel_reg_args: dict = None,
        abs_reg_args: dict = None,
    ):
        """
        Contrastive Regression Domain Adaptation (CoRe-DA).

        """
        super().__init__()

        # Feature extractor
        feat_args = feat_args or {}
        self.feature_extractor = I3DEncoder(**feat_args)

        # Relative Regressor
        rel_reg_args = rel_reg_args or {}
        self.rel_regressor = RelativeRegressor(**rel_reg_args)

        # Absolute Regressor
        abs_reg_args = abs_reg_args or {}
        self.abs_regressor = AbsoluteRegressor(**abs_reg_args)

    def _forward_train(self, x_S, x_T, x_ex):
        """
        Training forward pass.
        """
        f_S = self.feature_extractor(x_S)  # (B, T, f)
        f_T = self.feature_extractor(x_T)  # (B, T, f)
        f_ex = self.feature_extractor(x_ex)  # (B, T, f)

        dy_S_ex_hat = self.rel_regressor(f_S, f_ex)
        dy_T_ex_hat = self.rel_regressor(f_T, f_ex)
        y_S_hat = self.abs_regressor(f_S)
        y_T_hat = self.abs_regressor(f_T)
        y_ex_hat = self.abs_regressor(f_ex)

        return dy_S_ex_hat, dy_T_ex_hat, y_S_hat, y_T_hat, y_ex_hat

    def _forward_test(self, x_T, x_ex):
        """
        Testing forward pass.
        """
        f_T = self.feature_extractor(x_T)  # (B, T, f)
        f_ex = self.feature_extractor(x_ex)  # (B, T, f)

        dy_T_ex_hat = self.rel_regressor(f_T, f_ex)
        y_T_hat = self.abs_regressor(f_T)

        return dy_T_ex_hat, y_T_hat

    def forward(self, x_S=None, x_T=None, x_ex=None, mode="train"):

        if mode == "train":
            return self._forward_train(x_S, x_T, x_ex)
        elif mode == "test":
            return self._forward_test(x_T, x_ex)
        else:
            raise ValueError(f"Not valid mode {mode}")


class RelativeRegressor(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 256,
    ):

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, f_any, f_ex):

        # f_any shape: (B, T, f)
        f_any = f_any.mean(dim=1)  # (B, f)
        f_ex = f_ex.mean(dim=1)  # (B, f)
        out = torch.cat((f_any, f_ex), dim=1)  # (B, 2*f)
        return self.mlp(out)


class AbsoluteRegressor(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 256,
    ):

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, f_any):

        # f shape: (B, T, f)
        f_any = f_any.mean(dim=1)  # (B, f)
        return self.mlp(f_any)
