# FBD Configuration for BloodMNIST Plan 1
# Control whether clients can see model colors (True) or not (False)
TRANSPARENT_TO_CLIENT = False

MODEL_PARTS = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']

FBD_TRACE = {
    'AFA79': {'model_part': 'in_layer',  'color': 'M0', 'train_record': []},
    'AKY64': {'model_part': 'in_layer',  'color': 'M1', 'train_record': []},
    'ALT34': {'model_part': 'in_layer',  'color': 'M2', 'train_record': []},
    'AOC39': {'model_part': 'in_layer',  'color': 'M3', 'train_record': []},
    'ASN90': {'model_part': 'in_layer',  'color': 'M4', 'train_record': []},
    'AUV29': {'model_part': 'in_layer',  'color': 'M5', 'train_record': []},
    'BFF56': {'model_part': 'layer1',    'color': 'M0', 'train_record': []},
    'BVD88': {'model_part': 'layer1',    'color': 'M1', 'train_record': []},
    'BVP97': {'model_part': 'layer1',    'color': 'M2', 'train_record': []},
    'BWW19': {'model_part': 'layer1',    'color': 'M3', 'train_record': []},
    'BXG86': {'model_part': 'layer1',    'color': 'M4', 'train_record': []},
    'BYM04': {'model_part': 'layer1',    'color': 'M5', 'train_record': []},
    'CAA77': {'model_part': 'layer2',    'color': 'M0', 'train_record': []},
    'CGV29': {'model_part': 'layer2',    'color': 'M1', 'train_record': []},
    'CNF57': {'model_part': 'layer2',    'color': 'M2', 'train_record': []},
    'COO30': {'model_part': 'layer2',    'color': 'M3', 'train_record': []},
    'CPM83': {'model_part': 'layer2',    'color': 'M4', 'train_record': []},
    'CRZ52': {'model_part': 'layer2',    'color': 'M5', 'train_record': []},
    'DSC60': {'model_part': 'layer3',    'color': 'M0', 'train_record': []},
    'DQM27': {'model_part': 'layer3',    'color': 'M1', 'train_record': []},
    'DWJ41': {'model_part': 'layer3',    'color': 'M2', 'train_record': []},
    'DHK75': {'model_part': 'layer3',    'color': 'M3', 'train_record': []},
    'DPU42': {'model_part': 'layer3',    'color': 'M4', 'train_record': []},
    'DPX98': {'model_part': 'layer3',    'color': 'M5', 'train_record': []},
    'EJJ91': {'model_part': 'layer4',    'color': 'M0', 'train_record': []},
    'EVZ66': {'model_part': 'layer4',    'color': 'M1', 'train_record': []},
    'EGO46': {'model_part': 'layer4',    'color': 'M2', 'train_record': []},
    'EYT34': {'model_part': 'layer4',    'color': 'M3', 'train_record': []},
    'EVN11': {'model_part': 'layer4',    'color': 'M4', 'train_record': []},
    'EVN36': {'model_part': 'layer4',    'color': 'M5', 'train_record': []},
    'FXR03': {'model_part': 'out_layer', 'color': 'M0', 'train_record': []},
    'FPC91': {'model_part': 'out_layer', 'color': 'M1', 'train_record': []},
    'FBI78': {'model_part': 'out_layer', 'color': 'M2', 'train_record': []},
    'FGM06': {'model_part': 'out_layer', 'color': 'M3', 'train_record': []},
    'FWC09': {'model_part': 'out_layer', 'color': 'M4', 'train_record': []},
    'FSY05': {'model_part': 'out_layer', 'color': 'M5', 'train_record': []}
}

FBD_INFO = {
    # --- 1)  Model-to-client coverage -------------------------------
    # Each key is a model ID; its value is the list of clients that
    # participate in (and therefore update) that model.
    "models": {
        "M0": [0, 1, 2],
        "M1": [0, 3, 5],
        "M2": [0, 4, 5],
        "M3": [1, 3, 4],
        "M4": [1, 2, 3],
        "M5": [2, 4, 5],
    },

    # --- 2)  Client-to-model coverage -------------------------------
    # Each key is a client ID; its value is the three models in which
    # that client's update is replicated (r = 3 redundancy).
    "clients": {
        0: ["M0", "M1", "M2"],
        1: ["M0", "M3", "M4"],
        2: ["M0", "M4", "M5"],
        3: ["M1", "M3", "M4"],
        4: ["M2", "M3", "M5"],
        5: ["M1", "M2", "M5"],
    },

    # --- 3)  Training plan -----------------------------------------
    "training_plan": {
        # One global synchronisation per round; three rounds because
        # r = 3 (every client must touch each of its three models once).
        "rounds": 3,

        # Per-round schedule: {client_id: model_trained_that_round}.
        # Multiple clients may legitimately train the same model in
        # the same round; the server just aggregates their updates.
        "schedule": {
            0: {0: "M0", 1: "M3", 2: "M5", 3: "M4", 4: "M2", 5: "M1"},
            1: {0: "M1", 1: "M4", 2: "M0", 3: "M3", 4: "M5", 5: "M2"},
            2: {0: "M2", 1: "M0", 2: "M4", 3: "M1", 4: "M3", 5: "M5"},
        },

        # Minimal local work per communication round.
        "local_epochs_per_round": 1
    }
}

import json
from collections import defaultdict

# --- 1.  INPUT DICTS -------------------------------------------------
MODEL_PART_ORDER = ['in_layer', 'layer1', 'layer2',
                    'layer3', 'layer4', 'out_layer']

# --- 2.  DERIVED LOOK-UPS -------------------------------------------
part_rank = {p: i for i, p in enumerate(MODEL_PART_ORDER)}

# blocks belonging to each model, ordered by layer
model_to_blocks = {mid: [] for mid in FBD_INFO["models"]}
for bid, info in FBD_TRACE.items():
    model_to_blocks[info['color']].append(bid)
for blist in model_to_blocks.values():
    blist.sort(key=lambda b: part_rank[FBD_TRACE[b]['model_part']])

# --- 3.  HYPER-PARAMETERS -------------------------------------------
EPOCHS_PER_STAGE   = 10
# blocks_per_stage    = [6, 5, 3, 2, 1]   # 100 → ≈75 → 50 → ≈25 → ≈17 %
blocks_per_stage    = [6, 6, 6, 6, 6]
# blocks_per_stage    = [1, 1, 1, 1, 1]
OUTER_ROUNDS_TOTAL = EPOCHS_PER_STAGE * len(blocks_per_stage)
BLOCKS_PER_MODEL    = 6

# --- 4.  GENERATE SHIPPING / REQUEST PLANS ---------------------------
model_ptr      = {mid: 0 for mid in model_to_blocks}
shipping_plan  = defaultdict(dict)
request_plan   = defaultdict(dict)

for outer_round in range(OUTER_ROUNDS_TOTAL):          # 0-based index
    stage_idx  = outer_round // EPOCHS_PER_STAGE
    k_blocks   = blocks_per_stage[stage_idx]
    sched_idx  = outer_round % 3                      # micro-cycle 0/1/2
    schedule   = FBD_INFO["training_plan"]["schedule"][sched_idx]

    for client, model in schedule.items():
        ptr    = model_ptr[model]
        blist  = model_to_blocks[model]

        sel = blist[ptr:ptr + k_blocks]
        if len(sel) < k_blocks:                      # wrap-around
            sel += blist[:k_blocks - len(sel)]

        shipping_plan[outer_round + 1][client] = sel
        request_plan[outer_round + 1][client]  = sel
        model_ptr[model] = (ptr + k_blocks) % BLOCKS_PER_MODEL

# --- 5.  SAVE TO JSON -----------------------------------------------
with open("shipping_plan.json", "w") as f:
    json.dump({int(k): v for k, v in shipping_plan.items()}, f, indent=2)
with open("request_plan.json", "w") as f:
    json.dump({int(k): v for k, v in request_plan.items()}, f, indent=2)
