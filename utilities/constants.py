# Network cascading training method.
# SHARED mode keeps the same CSG-CRN network parameters for each cascade. One model produces all of the results.
# SEPARATE mode trains a new network for each cascade.
SHARED_PARAMS = "SHARED"
SEPARATE_PARAMS = "SEPARATE"
CASCADE_MODEL_MODES = [SHARED_PARAMS, SEPARATE_PARAMS]

# Loss sampling methods for near-surface samples.
# TARGET only includes samples near the target shape. 
# UNIFIED include samples near to either the target or reconstruction shape.
TARGET_SAMPLING = "TARGET"
UNIFIED_SAMPLING = "UNIFIED"
sampling_methods = [TARGET_SAMPLING, UNIFIED_SAMPLING]

# Number of uniform points to load for each required near-surface point.
NEAR_SURFACE_SAMPLE_FACTOR = 5
