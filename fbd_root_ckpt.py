"""
FBD Root Checkpoint Loader

This module provides functions to load ImageNet pretrained weights 
for ResNet18 and ResNet50 FBD models.
"""

import warnings
# Suppress torchvision pretrained warnings
warnings.filterwarnings("ignore", message=".*The parameter 'pretrained' is deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")

import torch
import torchvision.models as models
import logging
from typing import Dict, Any, Optional
from fbd_models import (
    get_resnet18_fbd_model, 
    get_resnet50_fbd_model,
    get_fbd_model
)


def load_imagenet_pretrained_resnet18() -> Dict[str, Any]:
    """
    Load ImageNet pretrained ResNet18 weights.
    
    Returns:
        dict: State dict with ImageNet pretrained weights
    """
    logging.info("Loading ImageNet pretrained ResNet18 weights...")
    pretrained_model = models.resnet18(weights='IMAGENET1K_V1')
    return pretrained_model.state_dict()


def load_imagenet_pretrained_resnet50() -> Dict[str, Any]:
    """
    Load ImageNet pretrained ResNet50 weights.
    
    Returns:
        dict: State dict with ImageNet pretrained weights
    """
    logging.info("Loading ImageNet pretrained ResNet50 weights...")
    pretrained_model = models.resnet50(weights='IMAGENET1K_V1')
    return pretrained_model.state_dict()


def adapt_pretrained_weights_to_fbd(pretrained_state_dict: Dict[str, Any], 
                                   target_model: torch.nn.Module,
                                   num_classes: int,
                                   norm_type: str = 'bn') -> Dict[str, Any]:
    """
    Adapt pretrained weights to FBD model structure with intelligent normalization handling.
    
    Args:
        pretrained_state_dict: ImageNet pretrained weights (always BatchNorm)
        target_model: FBD model to adapt weights for
        num_classes: Number of classes in target dataset
        norm_type: Target normalization type ('bn', 'in', 'ln')
        
    Returns:
        dict: Adapted state dict compatible with FBD model
    """
    adapted_state_dict = {}
    target_state_dict = target_model.state_dict()
    
    # Get the mapping of layers
    pretrained_keys = set(pretrained_state_dict.keys())
    target_keys = set(target_state_dict.keys())
    
    logging.info(f"Adapting pretrained weights: {len(pretrained_keys)} -> {len(target_keys)} parameters")
    logging.info(f"Target normalization type: {norm_type.upper()}")
    
    # Normalization layer patterns to handle differently
    bn_patterns = ['bn1', 'bn2', 'bn3', 'downsample.1']  # Common BN layer names
    norm_patterns = bn_patterns  # For IN/LN, these become different norm types
    
    # Statistics for logging
    matched_conv_keys = 0
    matched_norm_keys = 0
    skipped_norm_keys = 0
    adapted_fc_keys = 0
    
    # Copy compatible weights based on normalization type
    for target_key in target_keys:
        target_weight = target_state_dict[target_key]
        
        if target_key in pretrained_keys:
            pretrained_weight = pretrained_state_dict[target_key]
            
            # Check if this is a normalization layer
            is_norm_layer = any(pattern in target_key for pattern in norm_patterns)
            
            if is_norm_layer:
                # Handle normalization layers based on target norm type
                if norm_type == 'bn':
                    # BN -> BN: Direct copy if shapes match
                    if pretrained_weight.shape == target_weight.shape:
                        adapted_state_dict[target_key] = pretrained_weight
                        matched_norm_keys += 1
                    else:
                        logging.warning(f"BN shape mismatch for {target_key}: "
                                      f"pretrained {pretrained_weight.shape} vs target {target_weight.shape}")
                        adapted_state_dict[target_key] = target_weight
                else:
                    # BN -> IN/LN: Skip normalization weights, use random initialization
                    if 'running_mean' in target_key or 'running_var' in target_key:
                        # IN/LN don't have running statistics, skip these
                        logging.info(f"Skipping {target_key} (running stats not used in {norm_type.upper()})")
                        # Don't add to adapted_state_dict, will use target's initialization
                        skipped_norm_keys += 1
                        continue
                    elif 'weight' in target_key or 'bias' in target_key:
                        # For weight/bias in IN/LN, use target's random initialization
                        logging.info(f"Using random init for {target_key} ({norm_type.upper()} normalization)")
                        adapted_state_dict[target_key] = target_weight
                        skipped_norm_keys += 1
                    else:
                        adapted_state_dict[target_key] = target_weight
                        skipped_norm_keys += 1
            else:
                # Handle non-normalization layers (conv, fc, etc.)
                if pretrained_weight.shape == target_weight.shape:
                    adapted_state_dict[target_key] = pretrained_weight
                    if 'conv' in target_key:
                        matched_conv_keys += 1
                    elif 'fc' in target_key or 'linear' in target_key or 'out_layer' in target_key:
                        matched_conv_keys += 1  # Count as conv for simplicity
                else:
                    # Handle final layer (fc) with different number of classes
                    if 'fc' in target_key or 'out_layer' in target_key or 'linear' in target_key:
                        if num_classes != 1000:  # ImageNet has 1000 classes
                            logging.info(f"Adapting final layer {target_key}: "
                                       f"ImageNet({pretrained_weight.shape}) -> Target({target_weight.shape})")
                            # Use target model's initialized weights for different number of classes
                            adapted_state_dict[target_key] = target_weight
                            adapted_fc_keys += 1
                        else:
                            adapted_state_dict[target_key] = pretrained_weight
                            matched_conv_keys += 1
                    else:
                        logging.warning(f"Shape mismatch for {target_key}: "
                                      f"pretrained {pretrained_weight.shape} vs target {target_weight.shape}")
                        adapted_state_dict[target_key] = target_weight
        else:
            # Key not found in pretrained model, use target model's initialized weights
            adapted_state_dict[target_key] = target_weight
            
            # Log what we're using random initialization for
            if any(pattern in target_key for pattern in norm_patterns):
                skipped_norm_keys += 1
    
    # Comprehensive logging
    total_target_keys = len(target_keys)
    logging.info(f"Weight adaptation summary:")
    logging.info(f"  Convolutional weights loaded: {matched_conv_keys}")
    
    if norm_type == 'bn':
        logging.info(f"  Normalization weights loaded: {matched_norm_keys}")
        logging.info(f"  ✅ Full weight transfer (BN -> BN)")
    else:
        logging.info(f"  Normalization weights skipped: {skipped_norm_keys}")
        logging.info(f"  ✅ Hybrid transfer (Conv from ImageNet, {norm_type.upper()} random init)")
    
    if adapted_fc_keys > 0:
        logging.info(f"  Final layer adapted for {num_classes} classes: {adapted_fc_keys}")
    
    successful_transfers = len(adapted_state_dict)
    logging.info(f"  Total parameters: {successful_transfers}/{total_target_keys}")
    
    return adapted_state_dict


def load_fbd_model_with_imagenet_pretrained(architecture: str, 
                                           norm: str, 
                                           in_channels: int, 
                                           num_classes: int,
                                           device: str = 'cpu') -> torch.nn.Module:
    """
    Load FBD model with ImageNet pretrained weights.
    
    Args:
        architecture: 'resnet18' or 'resnet50'
        norm: Normalization type ('bn', 'in', 'ln')
        in_channels: Number of input channels
        num_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        torch.nn.Module: FBD model with pretrained weights loaded
    """
    # Create FBD model
    fbd_model = get_fbd_model(architecture, norm, in_channels, num_classes)
    fbd_model = fbd_model.to(device)
    
    # Load pretrained weights
    if architecture.lower() == 'resnet18':
        pretrained_weights = load_imagenet_pretrained_resnet18()
    elif architecture.lower() == 'resnet50':
        pretrained_weights = load_imagenet_pretrained_resnet50()
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    # Handle input channel mismatch (ImageNet uses 3 channels)
    if in_channels != 3:
        logging.info(f"Adapting first layer for {in_channels} input channels (ImageNet uses 3)")
        
        # Find the first convolution layer key
        first_conv_key = None
        for key in pretrained_weights.keys():
            if 'conv1.weight' in key or (key.startswith('conv1') and 'weight' in key):
                first_conv_key = key
                break
        
        if first_conv_key:
            original_conv1 = pretrained_weights[first_conv_key]  # Shape: [64, 3, 7, 7]
            
            if in_channels == 1:
                # Convert RGB to grayscale by averaging channels
                adapted_conv1 = original_conv1.mean(dim=1, keepdim=True)  # Shape: [64, 1, 7, 7]
                pretrained_weights[first_conv_key] = adapted_conv1
                logging.info(f"Converted first layer from 3 to 1 channel by averaging RGB channels")
            else:
                # For other channel numbers, replicate or truncate
                if in_channels < 3:
                    adapted_conv1 = original_conv1[:, :in_channels, :, :]
                else:
                    # Repeat channels if needed
                    repeats = (in_channels + 2) // 3  # Ceiling division
                    adapted_conv1 = original_conv1.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
                
                pretrained_weights[first_conv_key] = adapted_conv1
                logging.info(f"Adapted first layer from 3 to {in_channels} channels")
    
    # Adapt weights to FBD model structure
    adapted_weights = adapt_pretrained_weights_to_fbd(
        pretrained_weights, fbd_model, num_classes, norm
    )
    
    # Load adapted weights
    fbd_model.load_state_dict(adapted_weights, strict=False)
    
    logging.info(f"✅ Successfully loaded ImageNet pretrained {architecture.upper()} "
                f"into FBD model with {norm.upper()} normalization")
    
    return fbd_model


def get_pretrained_fbd_model(architecture: str, 
                           norm: str = 'bn', 
                           in_channels: int = 3, 
                           num_classes: int = 1000,
                           device: str = 'cpu',
                           use_pretrained: bool = True) -> torch.nn.Module:
    """
    Convenience function to get FBD model with optional pretrained weights.
    
    Args:
        architecture: 'resnet18' or 'resnet50'
        norm: Normalization type ('bn', 'in', 'ln')
        in_channels: Number of input channels
        num_classes: Number of output classes
        device: Device to load model on
        use_pretrained: Whether to load ImageNet pretrained weights
        
    Returns:
        torch.nn.Module: FBD model (with or without pretrained weights)
    """
    if use_pretrained:
        return load_fbd_model_with_imagenet_pretrained(
            architecture, norm, in_channels, num_classes, device
        )
    else:
        # Create model without pretrained weights
        model = get_fbd_model(architecture, norm, in_channels, num_classes)
        model = model.to(device)
        logging.info(f"✅ Created {architecture.upper()} FBD model without pretrained weights")
        return model


# Convenience functions for specific architectures
def get_pretrained_resnet18_fbd(norm: str = 'bn', 
                               in_channels: int = 3, 
                               num_classes: int = 1000,
                               device: str = 'cpu') -> torch.nn.Module:
    """Get ResNet18 FBD model with ImageNet pretrained weights."""
    return load_fbd_model_with_imagenet_pretrained(
        'resnet18', norm, in_channels, num_classes, device
    )


def get_pretrained_resnet50_fbd(norm: str = 'bn', 
                               in_channels: int = 3, 
                               num_classes: int = 1000,
                               device: str = 'cpu') -> torch.nn.Module:
    """Get ResNet50 FBD model with ImageNet pretrained weights."""
    return load_fbd_model_with_imagenet_pretrained(
        'resnet50', norm, in_channels, num_classes, device
    )


if __name__ == "__main__":
    # Test the functions
    logging.basicConfig(level=logging.INFO)
    
    print("Testing FBD Root Checkpoint Loader...")
    
    # Test all normalization types with ResNet18
    norm_types = ['bn', 'in', 'ln']
    
    for norm in norm_types:
        print(f"\n=== Testing ResNet18 FBD with {norm.upper()} + ImageNet Pretrained ===")
        model18 = get_pretrained_resnet18_fbd(
            norm=norm, 
            in_channels=1,  # For grayscale images like BloodMNIST
            num_classes=8,  # BloodMNIST has 8 classes
            device='cpu'
        )
        print(f"ResNet18 FBD model: {model18.__class__.__name__}")
    
    # Test all normalization types with ResNet50  
    for norm in norm_types:
        print(f"\n=== Testing ResNet50 FBD with {norm.upper()} + ImageNet Pretrained ===")
        model50 = get_pretrained_resnet50_fbd(
            norm=norm, 
            in_channels=1,  # For grayscale images like BloodMNIST
            num_classes=8,  # BloodMNIST has 8 classes
            device='cpu'
        )
        print(f"ResNet50 FBD model: {model50.__class__.__name__}")
    
    print("\n✅ All tests completed successfully!")
    print("\nSummary:")
    print("- BN models: Full weight transfer (conv + norm layers)")
    print("- IN/LN models: Hybrid transfer (conv from ImageNet, norm random init)")
    print("- All models: Final layer adapted for target classes") 