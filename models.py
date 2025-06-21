'''
Adapted from kuangliu/pytorch-cifar .
'''

import torch.nn as nn
import torch.nn.functional as F

MODEL_PARTS_RESNET18 = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
MODEL_PARTS_RESNET50 = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(in_channels, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)

class BasicBlock_IN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_IN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_LN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_LN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(1, planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(1, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18_FBD_BN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(ResNet18_FBD_BN, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.in_layer(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.out_layer(out)
        return out

    def load_from_dict(self, state_dicts):
        """
        Loads weights for specified layers from a dictionary of state_dicts.

        Args:
            state_dicts (dict): A dictionary where keys are the names of the
                                layers/blocks (e.g., 'in_layer', 'layer1') and
                                values are their corresponding state_dicts.
                                If a key matches a layer name in the model,
                                its weights will be replaced.
        """
        for layer_name, state_dict in state_dicts.items():
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                
                # Check if the state_dict has prefixed parameter names
                # If so, strip the prefix to match what the layer expects
                cleaned_state_dict = {}
                layer_prefix = layer_name + "."
                
                for param_name, param_tensor in state_dict.items():
                    if param_name.startswith(layer_prefix):
                        # Strip the layer prefix
                        clean_param_name = param_name[len(layer_prefix):]
                        cleaned_state_dict[clean_param_name] = param_tensor
                    else:
                        # Use as-is if no prefix
                        cleaned_state_dict[param_name] = param_tensor
                
                layer.load_state_dict(cleaned_state_dict)

    def send_for_dict(self, layer_names):
        """
        Extracts the state_dicts for a specified list of layers.

        Args:
            layer_names (list): A list of strings, where each string is the
                                name of a layer/block (e.g., 'in_layer', 'layer1').

        Returns:
            dict: A dictionary where keys are the layer names and values are
                  their corresponding state_dicts.
        """
        state_dicts_to_return = {}
        for layer_name in layer_names:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                state_dicts_to_return[layer_name] = layer.state_dict()
        return state_dicts_to_return

class ResNet18_FBD_IN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(ResNet18_FBD_IN, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64)
        )
        self.layer1 = nn.Sequential(
            BasicBlock_IN(64, 64, stride=1),
            BasicBlock_IN(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            BasicBlock_IN(64, 128, stride=2),
            BasicBlock_IN(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            BasicBlock_IN(128, 256, stride=2),
            BasicBlock_IN(256, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            BasicBlock_IN(256, 512, stride=2),
            BasicBlock_IN(512, 512, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.in_layer(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.out_layer(out)
        return out

    def load_from_dict(self, state_dicts):
        """
        Loads weights for specified layers from a dictionary of state_dicts.

        Args:
            state_dicts (dict): A dictionary where keys are the names of the
                                layers/blocks (e.g., 'in_layer', 'layer1') and
                                values are their corresponding state_dicts.
                                If a key matches a layer name in the model,
                                its weights will be replaced.
        """
        for layer_name, state_dict in state_dicts.items():
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                
                # Check if the state_dict has prefixed parameter names
                # If so, strip the prefix to match what the layer expects
                cleaned_state_dict = {}
                layer_prefix = layer_name + "."
                
                for param_name, param_tensor in state_dict.items():
                    if param_name.startswith(layer_prefix):
                        # Strip the layer prefix
                        clean_param_name = param_name[len(layer_prefix):]
                        cleaned_state_dict[clean_param_name] = param_tensor
                    else:
                        # Use as-is if no prefix
                        cleaned_state_dict[param_name] = param_tensor
                
                layer.load_state_dict(cleaned_state_dict)

    def send_for_dict(self, layer_names):
        """
        Extracts the state_dicts for a specified list of layers.

        Args:
            layer_names (list): A list of strings, where each string is the
                                name of a layer/block (e.g., 'in_layer', 'layer1').

        Returns:
            dict: A dictionary where keys are the layer names and values are
                  their corresponding state_dicts.
        """
        state_dicts_to_return = {}
        for layer_name in layer_names:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                state_dicts_to_return[layer_name] = layer.state_dict()
        return state_dicts_to_return

class ResNet18_FBD_LN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(ResNet18_FBD_LN, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, 64)
        )
        self.layer1 = nn.Sequential(
            BasicBlock_LN(64, 64, stride=1),
            BasicBlock_LN(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            BasicBlock_LN(64, 128, stride=2),
            BasicBlock_LN(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            BasicBlock_LN(128, 256, stride=2),
            BasicBlock_LN(256, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            BasicBlock_LN(256, 512, stride=2),
            BasicBlock_LN(512, 512, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.in_layer(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.out_layer(out)
        return out

    def load_from_dict(self, state_dicts):
        """
        Loads weights for specified layers from a dictionary of state_dicts.

        Args:
            state_dicts (dict): A dictionary where keys are the names of the
                                layers/blocks (e.g., 'in_layer', 'layer1') and
                                values are their corresponding state_dicts.
                                If a key matches a layer name in the model,
                                its weights will be replaced.
        """
        for layer_name, state_dict in state_dicts.items():
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                
                # Check if the state_dict has prefixed parameter names
                # If so, strip the prefix to match what the layer expects
                cleaned_state_dict = {}
                layer_prefix = layer_name + "."
                
                for param_name, param_tensor in state_dict.items():
                    if param_name.startswith(layer_prefix):
                        # Strip the layer prefix
                        clean_param_name = param_name[len(layer_prefix):]
                        cleaned_state_dict[clean_param_name] = param_tensor
                    else:
                        # Use as-is if no prefix
                        cleaned_state_dict[param_name] = param_tensor
                
                layer.load_state_dict(cleaned_state_dict)

    def send_for_dict(self, layer_names):
        """
        Extracts the state_dicts for a specified list of layers.

        Args:
            layer_names (list): A list of strings, where each string is the
                                name of a layer/block (e.g., 'in_layer', 'layer1').

        Returns:
            dict: A dictionary where keys are the layer names and values are
                  their corresponding state_dicts.
        """
        state_dicts_to_return = {}
        for layer_name in layer_names:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                state_dicts_to_return[layer_name] = layer.state_dict()
        return state_dicts_to_return

def ResNet50(in_channels, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)

class Bottleneck_IN(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_IN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck_LN(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_LN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(1, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(1, planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(1, self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet50_FBD_BN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(ResNet50_FBD_BN, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, stride=1),
            Bottleneck(256, 64, stride=1),
            Bottleneck(256, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, stride=2),
            Bottleneck(512, 128, stride=1),
            Bottleneck(512, 128, stride=1),
            Bottleneck(512, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(512, 256, stride=2),
            Bottleneck(1024, 256, stride=1),
            Bottleneck(1024, 256, stride=1),
            Bottleneck(1024, 256, stride=1),
            Bottleneck(1024, 256, stride=1),
            Bottleneck(1024, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, stride=2),
            Bottleneck(2048, 512, stride=1),
            Bottleneck(2048, 512, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_layer = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = F.relu(self.in_layer(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.out_layer(out)
        return out

    def load_from_dict(self, state_dicts):
        """
        Loads weights for specified layers from a dictionary of state_dicts.

        Args:
            state_dicts (dict): A dictionary where keys are the names of the
                                layers/blocks (e.g., 'in_layer', 'layer1') and
                                values are their corresponding state_dicts.
                                If a key matches a layer name in the model,
                                its weights will be replaced.
        """
        for layer_name, state_dict in state_dicts.items():
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                
                # Check if the state_dict has prefixed parameter names
                # If so, strip the prefix to match what the layer expects
                cleaned_state_dict = {}
                layer_prefix = layer_name + "."
                
                for param_name, param_tensor in state_dict.items():
                    if param_name.startswith(layer_prefix):
                        # Strip the layer prefix
                        clean_param_name = param_name[len(layer_prefix):]
                        cleaned_state_dict[clean_param_name] = param_tensor
                    else:
                        # Use as-is if no prefix
                        cleaned_state_dict[param_name] = param_tensor
                
                layer.load_state_dict(cleaned_state_dict)

    def send_for_dict(self, layer_names):
        """
        Extracts the state_dicts for a specified list of layers.

        Args:
            layer_names (list): A list of strings, where each string is the
                                name of a layer/block (e.g., 'in_layer', 'layer1').

        Returns:
            dict: A dictionary where keys are the layer names and values are
                  their corresponding state_dicts.
        """
        state_dicts_to_return = {}
        for layer_name in layer_names:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                state_dicts_to_return[layer_name] = layer.state_dict()
        return state_dicts_to_return

class ResNet50_FBD_IN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(ResNet50_FBD_IN, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64)
        )
        self.layer1 = nn.Sequential(
            Bottleneck_IN(64, 64, stride=1),
            Bottleneck_IN(256, 64, stride=1),
            Bottleneck_IN(256, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            Bottleneck_IN(256, 128, stride=2),
            Bottleneck_IN(512, 128, stride=1),
            Bottleneck_IN(512, 128, stride=1),
            Bottleneck_IN(512, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            Bottleneck_IN(512, 256, stride=2),
            Bottleneck_IN(1024, 256, stride=1),
            Bottleneck_IN(1024, 256, stride=1),
            Bottleneck_IN(1024, 256, stride=1),
            Bottleneck_IN(1024, 256, stride=1),
            Bottleneck_IN(1024, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            Bottleneck_IN(1024, 512, stride=2),
            Bottleneck_IN(2048, 512, stride=1),
            Bottleneck_IN(2048, 512, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_layer = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = F.relu(self.in_layer(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.out_layer(out)
        return out

    def load_from_dict(self, state_dicts):
        """
        Loads weights for specified layers from a dictionary of state_dicts.

        Args:
            state_dicts (dict): A dictionary where keys are the names of the
                                layers/blocks (e.g., 'in_layer', 'layer1') and
                                values are their corresponding state_dicts.
                                If a key matches a layer name in the model,
                                its weights will be replaced.
        """
        for layer_name, state_dict in state_dicts.items():
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                
                # Check if the state_dict has prefixed parameter names
                # If so, strip the prefix to match what the layer expects
                cleaned_state_dict = {}
                layer_prefix = layer_name + "."
                
                for param_name, param_tensor in state_dict.items():
                    if param_name.startswith(layer_prefix):
                        # Strip the layer prefix
                        clean_param_name = param_name[len(layer_prefix):]
                        cleaned_state_dict[clean_param_name] = param_tensor
                    else:
                        # Use as-is if no prefix
                        cleaned_state_dict[param_name] = param_tensor
                
                layer.load_state_dict(cleaned_state_dict)

    def send_for_dict(self, layer_names):
        """
        Extracts the state_dicts for a specified list of layers.

        Args:
            layer_names (list): A list of strings, where each string is the
                                name of a layer/block (e.g., 'in_layer', 'layer1').

        Returns:
            dict: A dictionary where keys are the layer names and values are
                  their corresponding state_dicts.
        """
        state_dicts_to_return = {}
        for layer_name in layer_names:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                state_dicts_to_return[layer_name] = layer.state_dict()
        return state_dicts_to_return

class ResNet50_FBD_LN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(ResNet50_FBD_LN, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, 64)
        )
        self.layer1 = nn.Sequential(
            Bottleneck_LN(64, 64, stride=1),
            Bottleneck_LN(256, 64, stride=1),
            Bottleneck_LN(256, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            Bottleneck_LN(256, 128, stride=2),
            Bottleneck_LN(512, 128, stride=1),
            Bottleneck_LN(512, 128, stride=1),
            Bottleneck_LN(512, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            Bottleneck_LN(512, 256, stride=2),
            Bottleneck_LN(1024, 256, stride=1),
            Bottleneck_LN(1024, 256, stride=1),
            Bottleneck_LN(1024, 256, stride=1),
            Bottleneck_LN(1024, 256, stride=1),
            Bottleneck_LN(1024, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            Bottleneck_LN(1024, 512, stride=2),
            Bottleneck_LN(2048, 512, stride=1),
            Bottleneck_LN(2048, 512, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_layer = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = F.relu(self.in_layer(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.out_layer(out)
        return out

    def load_from_dict(self, state_dicts):
        """
        Loads weights for specified layers from a dictionary of state_dicts.

        Args:
            state_dicts (dict): A dictionary where keys are the names of the
                                layers/blocks (e.g., 'in_layer', 'layer1') and
                                values are their corresponding state_dicts.
                                If a key matches a layer name in the model,
                                its weights will be replaced.
        """
        for layer_name, state_dict in state_dicts.items():
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                
                # Check if the state_dict has prefixed parameter names
                # If so, strip the prefix to match what the layer expects
                cleaned_state_dict = {}
                layer_prefix = layer_name + "."
                
                for param_name, param_tensor in state_dict.items():
                    if param_name.startswith(layer_prefix):
                        # Strip the layer prefix
                        clean_param_name = param_name[len(layer_prefix):]
                        cleaned_state_dict[clean_param_name] = param_tensor
                    else:
                        # Use as-is if no prefix
                        cleaned_state_dict[param_name] = param_tensor
                
                layer.load_state_dict(cleaned_state_dict)

    def send_for_dict(self, layer_names):
        """
        Extracts the state_dicts for a specified list of layers.

        Args:
            layer_names (list): A list of strings, where each string is the
                                name of a layer/block (e.g., 'in_layer', 'layer1').

        Returns:
            dict: A dictionary where keys are the layer names and values are
                  their corresponding state_dicts.
        """
        state_dicts_to_return = {}
        for layer_name in layer_names:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                state_dicts_to_return[layer_name] = layer.state_dict()
        return state_dicts_to_return

# Helper functions for FBD model selection

def get_resnet18_fbd_model(norm: str, in_channels: int, num_classes: int):
    """
    Get the appropriate ResNet18 FBD model based on normalization type.
    
    Args:
        norm (str): Normalization type - 'bn', 'in', or 'ln'
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        
    Returns:
        nn.Module: ResNet18 FBD model with specified normalization
    """
    if norm == 'bn':
        return ResNet18_FBD_BN(in_channels=in_channels, num_classes=num_classes)
    elif norm == 'in':
        return ResNet18_FBD_IN(in_channels=in_channels, num_classes=num_classes)
    elif norm == 'ln':
        return ResNet18_FBD_LN(in_channels=in_channels, num_classes=num_classes)
    else:
        # Default to batch normalization if norm type is not specified or unknown
        return ResNet18_FBD_BN(in_channels=in_channels, num_classes=num_classes)

def get_resnet50_fbd_model(norm: str, in_channels: int, num_classes: int):
    """
    Get the appropriate ResNet50 FBD model based on normalization type.
    
    Args:
        norm (str): Normalization type - 'bn', 'in', or 'ln'
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        
    Returns:
        nn.Module: ResNet50 FBD model with specified normalization
    """
    if norm == 'bn':
        return ResNet50_FBD_BN(in_channels=in_channels, num_classes=num_classes)
    elif norm == 'in':
        return ResNet50_FBD_IN(in_channels=in_channels, num_classes=num_classes)
    elif norm == 'ln':
        return ResNet50_FBD_LN(in_channels=in_channels, num_classes=num_classes)
    else:
        # Default to batch normalization if norm type is not specified or unknown
        return ResNet50_FBD_BN(in_channels=in_channels, num_classes=num_classes)

def get_fbd_model(architecture: str, norm: str, in_channels: int, num_classes: int):
    """
    Get the appropriate FBD model based on architecture and normalization type.
    
    Args:
        architecture (str): Model architecture - 'resnet18' or 'resnet50'
        norm (str): Normalization type - 'bn', 'in', or 'ln'
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        
    Returns:
        nn.Module: FBD model with specified architecture and normalization
        
    Raises:
        ValueError: If architecture is not supported
    """
    if architecture.lower() == 'resnet18':
        return get_resnet18_fbd_model(norm, in_channels, num_classes)
    elif architecture.lower() == 'resnet50':
        return get_resnet50_fbd_model(norm, in_channels, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: 'resnet18', 'resnet50'")

def get_model_parts(architecture: str):
    """
    Get the model parts list for the specified architecture.
    
    Args:
        architecture (str): Model architecture - 'resnet18' or 'resnet50'
        
    Returns:
        list: List of model part names
        
    Raises:
        ValueError: If architecture is not supported
    """
    if architecture.lower() == 'resnet18':
        return MODEL_PARTS_RESNET18
    elif architecture.lower() == 'resnet50':
        return MODEL_PARTS_RESNET50
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: 'resnet18', 'resnet50'")

def get_model_info(architecture: str, norm: str):
    """
    Get comprehensive model information including class name and description.
    
    Args:
        architecture (str): Model architecture - 'resnet18' or 'resnet50'
        norm (str): Normalization type - 'bn', 'in', or 'ln'
        
    Returns:
        dict: Dictionary containing model information
    """
    arch_lower = architecture.lower()
    norm_lower = norm.lower()
    
    # Normalization descriptions
    norm_descriptions = {
        'bn': 'Batch Normalization',
        'in': 'Instance Normalization', 
        'ln': 'Layer Normalization'
    }
    
    # Class name mapping
    class_names = {
        'resnet18': {
            'bn': 'ResNet18_FBD_BN',
            'in': 'ResNet18_FBD_IN',
            'ln': 'ResNet18_FBD_LN'
        },
        'resnet50': {
            'bn': 'ResNet50_FBD_BN',
            'in': 'ResNet50_FBD_IN', 
            'ln': 'ResNet50_FBD_LN'
        }
    }
    
    if arch_lower not in class_names:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: 'resnet18', 'resnet50'")
    
    if norm_lower not in norm_descriptions:
        # Default to batch normalization
        norm_lower = 'bn'
    
    return {
        'architecture': arch_lower,
        'normalization': norm_lower,
        'class_name': class_names[arch_lower][norm_lower],
        'description': f"{architecture.upper()} FBD with {norm_descriptions[norm_lower]}",
        'model_parts': get_model_parts(architecture)
    } 