"""Hyperparameters for the PPMI/SVD book embedding."""
MAX_BOOKS_PER_SUBJECT = 50
WINDOW = 10             # context words on each side
MIN_COUNT = 10          # vocabulary threshold, whole book
MIN_TOKEN_SIZE = 3      # shorter tokens are dropped
VECTOR_SIZE = 100       # SVD dimensionality
GAMMA = 0.5             # eigenvalue weighting, w = U * S**gamma
ALPHA = 0.0             # PMI shift subtracted before clipping
CDS = 0.75              # context distribution smoothing exponent
SOLVER = "arpack"       # 'arpack' (accurate) or 'propack' (~2x faster)
V0_SEED = 0             # Lanczos start vector; affects nothing measurable
