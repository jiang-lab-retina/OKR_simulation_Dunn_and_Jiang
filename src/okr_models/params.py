"""Model parameters as stated in the paper (Methods, MODELING; Fig 4 and Fig S5 legends).

All lengths are in pixels of the 1024 x 1024 px model images; speeds in pixels per frame.
"""

IMAGE_SIZE = 1024                      # px, square images and movies

# Moving dots (Methods, Stimuli)
DOTS = dict(
    seed=12345,
    num_frames=9,
    fps=90,
    num_particles=300,
    particle_diameter_px=39,
    velocity_px_per_sec=1200.0,        # 1200 / 90 = 13.33 px/frame upward (ground-truth speed)
    visible_ms_range=(150, 200),       # each dot is visible for 150-200 ms ...
    invisible_ms_range=(150, 200),     # ... then invisible for 150-200 ms, then reappears elsewhere
    initial_timer_ms_range=(0, 200),
    background=32,                     # uint8 grey levels; movies are divided by 255 before analysis
    foreground=255,
)
GT_SPEED_PX_PER_FRAME = DOTS["velocity_px_per_sec"] / DOTS["fps"]   # 13.33

# Drifting sinusoidal gratings (Fig 4a, 4e): 2-32 cycles per image width, drifting rightward
GRATING_CYCLES = (2, 4, 8, 16, 32)

# Direction ground truth (Methods, Direction estimation)
GT_DIRECTION_DEG = {"dots": 90.0, "grating": 180.0}

# Scotoma masks: 51 independent realisations x 20 levels (0, 5, ..., 95 %)
MASK_SEEDS = tuple(range(100, 151))
SCOTOMA_LEVELS_PCT = tuple(float(5 * i) for i in range(20))
HOLE_RADIUS_PX = 40                    # each scotoma is a disc of radius 40 px
MASK_TOLERANCE_PCT = 0.5               # realised coverage within 0.5 percentage points of the level
MASK_MAX_HOLES_FACTOR = 3.0

# Direction posterior (Methods, Direction estimation)
POSTERIOR_SIGMA = 0.05
ANGLE_STEP_DEG = 0.25

# Lucas-Kanade integration windows: radius and stride in px
WINDOWS = {
    "r3": (3, 2),          # Fig 4a-d, Fig S5
    "rw120": (120, 120),   # Fig 4e   (R_w = 120 px, 240 px window)
    "rw105": (105, 110),   # Fig 4f-h (R_w = 105 px, 210 px window)
}

# Optotype E (Methods, Orientation discrimination)
OPTOTYPE = dict(
    font_size=800,             # height of the E in px; stroke width = font_size // 5 = 160 px
    background=0.5,            # grey background; scotomas take the background value
    foreground=1.0,
    orientations=(0, 90, 180, 270),
    softmax_temperature=0.05,  # confidence = softmax of the four template correlations
)
OPTOTYPE_SIGMAS = (0.0, 10.0, 100.0)
