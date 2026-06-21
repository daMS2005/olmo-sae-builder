"""Loss functions for the simple PyTorch SAE in model.py.

The SAE is trained to reconstruct the input activation while keeping latent
activations small/sparse. The reconstruction term measures error in input space,
and the L1 term charges the model for using latent activation mass.
"""

import torch


def autoencoder_loss(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    latent_activations: torch.Tensor,
    l1_weight: float,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param l1_weight: weight of L1 loss
    :return: loss (shape: [1])
    """
    # Add the reconstruction penalty and sparsity penalty into one scalar loss.
    # l1_weight controls how expensive latent activation is relative to
    # reconstruction error.
    return (
        normalized_mean_squared_error(reconstruction, original_input)
        + normalized_L1_loss(latent_activations, original_input) * l1_weight
    )


def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error (shape: [1])
    """
    # For each example, divide average squared reconstruction error by average
    # squared input value. This makes the error relative to that example's input
    # scale before averaging over the batch.
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1) / (original_input**2).mean(dim=1)
    ).mean()


def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized L1 loss (shape: [1])
    """
    # Sum absolute latent values for each example, then divide by the input norm
    # so the sparsity cost is measured relative to the activation being encoded.
    return (latent_activations.abs().sum(dim=1) / original_input.norm(dim=1)).mean()
