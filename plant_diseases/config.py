"""
Shared constants used across the entire project.

WHY DO WE NEED THIS?
Every photo is made of pixels. Each pixel has three colour values:
Red, Green, and Blue (RGB), each between 0 and 255.

Neural networks learn faster when numbers are small and centred around
zero instead of 0-255. To get there we subtract the average colour and
divide by the spread. These two lists are the average and spread measured
across 1.2 million photos (the ImageNet dataset) that our pretrained
network was originally trained on. We MUST use the same numbers so the
network sees colours the way it expects.
"""

# Average Red, Green, Blue across 1.2 million ImageNet photos.
IMAGENET_MEAN = [0.485, 0.456, 0.406]

# Spread (standard deviation) of Red, Green, Blue across ImageNet.
IMAGENET_STD = [0.229, 0.224, 0.225]

# The pixel size EfficientNet-B0 expects as input.
INPUT_SIZE = 224

# When resizing before a centre crop, resize to this first.
RESIZE_SIZE = 256

# Dropout probability for the classifier head.
DROPOUT_RATE = 0.3
