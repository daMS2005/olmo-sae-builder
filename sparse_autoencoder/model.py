"""Small, readable sparse autoencoder building blocks.

This file is the best place to start if you are learning the codebase.

Main tensors:
    x                   [batch, n_inputs]   input activation vectors
    latents_pre_act     [batch, n_latents]  feature scores before sparsity
    latents             [batch, n_latents]  sparse feature activations
    recons              [batch, n_inputs]   reconstructed activation vectors
"""

from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Layer-normalize each activation row, while keeping stats to undo it.

    Each row is normalized independently because the SAE may be trained on
    normalized activations. The mean and standard deviation are returned so
    decode() can turn a normalized reconstruction back into the original scale.
    """
    # mu is [batch, 1]: one mean per activation vector, kept as a column so it
    # broadcasts cleanly across the feature dimension.
    mu = x.mean(dim=-1, keepdim=True)

    # Subtracting mu centers each row, which is the first half of layer norm.
    x = x - mu

    # std is [batch, 1]: one scale value per centered activation vector.
    std = x.std(dim=-1, keepdim=True)

    # Dividing by std makes each row unit-scale; eps keeps the division stable
    # for rows with tiny variance.
    x = x / (std + eps)

    # Returning mu/std makes the inverse transform explicit:
    # original = normalized * std + mu.
    return x, mu, std


class Autoencoder(nn.Module):
    """Sparse autoencoder

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self, n_latents: int, n_inputs: int, activation: Callable = nn.ReLU(), tied: bool = False,
        normalize: bool = False
    ) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
        :param activation: activation function
        :param tied: whether to tie the encoder and decoder weights
        """
        super().__init__()

        # pre_bias is a learned [n_inputs] baseline: the encoder scores
        # x - pre_bias, and the decoder adds the same baseline back so
        # reconstructions live in the original activation space.
        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))

        # encoder.weight is [n_latents, n_inputs]. Each row detects one latent
        # feature by dotting with the centered input; F.linear implements this as
        # x @ encoder.weight.T.
        self.encoder: nn.Module = nn.Linear(n_inputs, n_latents, bias=False)

        # latent_bias is a learned [n_latents] offset added before the sparsifying
        # activation, so each feature can have its own activation threshold.
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation

        # The decoder turns sparse latents back into [n_inputs] vectors. Tied
        # mode reuses encoder.weight.T; untied mode learns a separate matrix.
        if tied:
            self.decoder: nn.Linear | TiedTranspose = TiedTranspose(self.encoder)
        else:
            # decoder.weight is [n_inputs, n_latents], so column i is the
            # input-space direction contributed by latent i.
            self.decoder = nn.Linear(n_latents, n_inputs, bias=False)

        # normalize marks checkpoints that expect LN inputs; preprocess() applies
        # LN and decode() uses the stored stats to reverse it.
        self.normalize = normalize

        self.stats_last_nonzero: torch.Tensor
        self.latents_activation_frequency: torch.Tensor
        self.latents_mean_square: torch.Tensor

        # These buffers track per-latent statistics across forward passes. They
        # move with the module and save in checkpoints, but optimizers do not
        # update them as learned parameters.
        self.register_buffer("stats_last_nonzero", torch.zeros(n_latents, dtype=torch.long))
        self.register_buffer(
            "latents_activation_frequency", torch.ones(n_latents, dtype=torch.float)
        )
        self.register_buffer("latents_mean_square", torch.zeros(n_latents, dtype=torch.float))

    def encode_pre_act(self, x: torch.Tensor, latent_slice: slice = slice(None)) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """
        # Center every [n_inputs] activation row around pre_bias before feature
        # scoring, using broadcasting over the batch dimension.
        x = x - self.pre_bias

        # F.linear computes x @ weight.T + bias. Slicing the weight/bias lets
        # callers score only selected latent features, producing
        # [batch, selected_latents].
        latents_pre_act = F.linear(
            x, self.encoder.weight[latent_slice], self.latent_bias[latent_slice]
        )
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply the input transform expected by this checkpoint."""
        if not self.normalize:
            return x, dict()
        # When normalization is enabled, keep mu/std in info because decode()
        # needs them to put reconstructions back on the original scale.
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        # encode() exposes the sparse code without reconstructing. It returns
        # info alongside latents so decode(latents, info) can undo preprocessing.
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(self, latents: torch.Tensor, info: dict[str, Any] | None = None) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        # decoder(latents) sums the active decoder directions into
        # [batch, n_inputs], then pre_bias restores the baseline that encode()
        # subtracted.
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            # The decoder output is still in normalized coordinates, so reverse
            # LN with the exact row-wise stats saved by preprocess().
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """
        # forward() returns all three major tensors so training/analysis can use
        # raw scores, sparse activations, and reconstructions from one pass.
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        # stats_last_nonzero[i] counts how many forward calls have passed since
        # latent i fired. The all(dim=0) mask resets features used by this batch
        # and increments features that stayed zero.
        self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
        self.stats_last_nonzero += 1

        return latents_pre_act, latents, recons

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> "Autoencoder":
        """Recreate an Autoencoder from saved weights and activation metadata."""
        n_latents, d_model = state_dict["encoder.weight"].shape

        # Retrieve activation
        activation_class_name = state_dict.pop("activation", "ReLU")
        activation_class = ACTIVATIONS_CLASSES.get(activation_class_name, nn.ReLU)

        # The saved activation name tells us both which activation object to
        # rebuild and, for this checkpoint family, whether LN preprocessing is
        # expected.
        normalize = activation_class_name == "TopK"
        activation_state_dict = state_dict.pop("activation_state_dict", {})

        # Rebuild the activation module before loading tensor weights, because
        # TopK stores non-parameter settings like k in activation_state_dict.
        if hasattr(activation_class, "from_state_dict"):
            activation = activation_class.from_state_dict(
                activation_state_dict, strict=strict
            )
        else:
            activation = activation_class()
            if hasattr(activation, "load_state_dict"):
                activation.load_state_dict(activation_state_dict, strict=strict)

        autoencoder = cls(n_latents, d_model, activation=activation, normalize=normalize)
        # Load the remaining tensors into the freshly constructed module.
        autoencoder.load_state_dict(state_dict, strict=strict)
        return autoencoder

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save weights plus the activation metadata needed to reload them."""
        sd = super().state_dict(destination, prefix, keep_vars)
        sd[prefix + "activation"] = self.activation.__class__.__name__
        if hasattr(self.activation, "state_dict"):
            sd[prefix + "activation_state_dict"] = self.activation.state_dict()
        return sd


class TiedTranspose(nn.Module):
    """Decoder module that reuses encoder.weight.T instead of owning weights."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        # F.linear wants [out_features, in_features]. Transposing the encoder
        # weight turns [n_latents, n_inputs] into decoder-shaped
        # [n_inputs, n_latents].
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        # Expose a decoder-shaped view so code reading decoder.weight sees
        # [n_inputs, n_latents].
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


class TopK(nn.Module):
    """Sparse activation that keeps the k largest feature scores per row."""

    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # topk returns the selected scores and their latent indices, both
        # [batch, k], so the module knows what to keep and where to put it back.
        topk = torch.topk(x, k=self.k, dim=-1)

        # The post-activation transforms only the winning scores; the other
        # latent dimensions will remain exactly zero.
        values = self.postact_fn(topk.values)

        # Start from a zero tensor so non-winning latents are explicitly absent.
        result = torch.zeros_like(x)

        # scatter_ writes the selected values back to their original latent
        # positions, converting compact top-k output back to [batch, n_latents].
        result.scatter_(-1, topk.indices, values)
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # k and postact_fn are Python settings rather than Parameters, so store
        # them as metadata beside the normal PyTorch state dict.
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict.update({prefix + "k": self.k, prefix + "postact_fn": self.postact_fn.__class__.__name__})
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], strict: bool = True) -> "TopK":
        # Recreate the same TopK module from the metadata saved by state_dict().
        k = state_dict["k"]
        postact_fn = ACTIVATIONS_CLASSES[state_dict["postact_fn"]]()
        return cls(k=k, postact_fn=postact_fn)


ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}
