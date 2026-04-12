"""
Plant disease classification toolkit.

This package organises everything needed to train a neural network that
looks at photos of plant leaves and tells you which disease the plant has
(or if it is healthy).

Think of it like a toolbox:
  - config      : shared settings everyone uses (like paint colours)
  - device      : picks the fastest computer chip available
  - transforms  : prepares photos so the network can read them
  - data        : loads thousands of photos from folders on disk
  - model       : builds and manages the neural network itself
  - trainer     : runs the learning loop (show photos, learn, repeat)
  - history     : keeps a diary of how training is going
  - plotting    : draws charts from that diary
"""
