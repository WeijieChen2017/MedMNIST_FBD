# FBD Configuration Settings for BloodMNIST Plan 1
# This file contains ONLY configuration parameters - no executable code

# --- BASIC SETTINGS ---
TRANSPARENT_TO_CLIENT = False
MODEL_PARTS = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
MODEL_PART_ORDER = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']

# --- REGULARIZER CONFIGURATION ---
# REGULARIZER_PARAMS = {
#     "type": "consistency loss",
#     "distance_type": "L2", 
#     "coefficient": 0.1,
# }

# Alternative regularizer (commented out):
REGULARIZER_PARAMS = {
    "type": "weights",
    "distance_type": "L2",
    "coefficient": 0.01,
}

# --- TRAINING HYPER-PARAMETERS ---
EPOCHS_PER_STAGE = 20
BLOCKS_PER_STAGE = [6, 6, 6, 6, 6]  # Currently ignored - see documentation
ENSEMBLE_SIZE = 24
ENSEMBLE_COLORS = ['M1', 'M2']

# --- DERIVED PARAMETERS ---
OUTER_ROUNDS_TOTAL = EPOCHS_PER_STAGE * len(BLOCKS_PER_STAGE)  # = 100
BLOCKS_PER_MODEL = len(MODEL_PARTS)  # = 6

# --- BLOCK TRACE MAPPING ---
# Maps each block ID to its model part, color, and training record
FBD_TRACE = {
    # M0 blocks
    'AFA79': {'model_part': 'in_layer',  'color': 'M0', 'train_record': []},
    'BFF56': {'model_part': 'layer1',    'color': 'M0', 'train_record': []},
    'CAA77': {'model_part': 'layer2',    'color': 'M0', 'train_record': []},
    'DSC60': {'model_part': 'layer3',    'color': 'M0', 'train_record': []},
    'EJJ91': {'model_part': 'layer4',    'color': 'M0', 'train_record': []},
    'FXR03': {'model_part': 'out_layer', 'color': 'M0', 'train_record': []},
    
    # M1 blocks
    'AKY64': {'model_part': 'in_layer',  'color': 'M1', 'train_record': []},
    'BVD88': {'model_part': 'layer1',    'color': 'M1', 'train_record': []},
    'CGV29': {'model_part': 'layer2',    'color': 'M1', 'train_record': []},
    'DQM27': {'model_part': 'layer3',    'color': 'M1', 'train_record': []},
    'EVZ66': {'model_part': 'layer4',    'color': 'M1', 'train_record': []},
    'FPC91': {'model_part': 'out_layer', 'color': 'M1', 'train_record': []},
    
    # M2 blocks
    'ALT34': {'model_part': 'in_layer',  'color': 'M2', 'train_record': []},
    'BVP97': {'model_part': 'layer1',    'color': 'M2', 'train_record': []},
    'CNF57': {'model_part': 'layer2',    'color': 'M2', 'train_record': []},
    'DWJ41': {'model_part': 'layer3',    'color': 'M2', 'train_record': []},
    'EGO46': {'model_part': 'layer4',    'color': 'M2', 'train_record': []},
    'FBI78': {'model_part': 'out_layer', 'color': 'M2', 'train_record': []},
    
    # M3 blocks
    'AOC39': {'model_part': 'in_layer',  'color': 'M3', 'train_record': []},
    'BWW19': {'model_part': 'layer1',    'color': 'M3', 'train_record': []},
    'COO30': {'model_part': 'layer2',    'color': 'M3', 'train_record': []},
    'DHK75': {'model_part': 'layer3',    'color': 'M3', 'train_record': []},
    'EYT34': {'model_part': 'layer4',    'color': 'M3', 'train_record': []},
    'FGM06': {'model_part': 'out_layer', 'color': 'M3', 'train_record': []},
    
    # M4 blocks
    'ASN90': {'model_part': 'in_layer',  'color': 'M4', 'train_record': []},
    'BXG86': {'model_part': 'layer1',    'color': 'M4', 'train_record': []},
    'CPM83': {'model_part': 'layer2',    'color': 'M4', 'train_record': []},
    'DPU42': {'model_part': 'layer3',    'color': 'M4', 'train_record': []},
    'EVN11': {'model_part': 'layer4',    'color': 'M4', 'train_record': []},
    'FWC09': {'model_part': 'out_layer', 'color': 'M4', 'train_record': []},
    
    # M5 blocks
    'AUV29': {'model_part': 'in_layer',  'color': 'M5', 'train_record': []},
    'BYM04': {'model_part': 'layer1',    'color': 'M5', 'train_record': []},
    'CRZ52': {'model_part': 'layer2',    'color': 'M5', 'train_record': []},
    'DPX98': {'model_part': 'layer3',    'color': 'M5', 'train_record': []},
    'EVN36': {'model_part': 'layer4',    'color': 'M5', 'train_record': []},
    'FSY05': {'model_part': 'out_layer', 'color': 'M5', 'train_record': []}
}

# --- CLIENT-MODEL ASSIGNMENT ---
FBD_INFO = {
    # Model-to-client coverage: which clients train each model
    "models": {
        "M0": [0, 1, 2],
        "M1": [0, 3, 5],
        "M2": [0, 4, 5],
        "M3": [1, 3, 4],
        "M4": [1, 2, 3],
        "M5": [2, 4, 5],
    },

    # Client-to-model coverage: which models each client trains (r=3 redundancy)
    "clients": {
        0: ["M0", "M1", "M2"],
        1: ["M0", "M3", "M4"],
        2: ["M0", "M4", "M5"],
        3: ["M1", "M3", "M4"],
        4: ["M2", "M3", "M5"],
        5: ["M1", "M2", "M5"],
    },

    # Training schedule: 3-round rotation
    "training_plan": {
        "rounds": 3,
        "schedule": {
            # Round 0: client -> model assignments
            0: {0: "M0", 1: "M3", 2: "M5", 3: "M4", 4: "M2", 5: "M1"},
            # Round 1: client -> model assignments  
            1: {0: "M1", 1: "M4", 2: "M0", 3: "M3", 4: "M5", 5: "M2"},
            # Round 2: client -> model assignments
            2: {0: "M2", 1: "M0", 2: "M4", 3: "M1", 4: "M3", 5: "M5"},
        },
        "local_epochs_per_round": 1
    }
} 