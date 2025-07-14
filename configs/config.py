import torch


class Config:
    # Dataset parameters
    DATASET_URL = "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"
    DATASET_FILE_NAME = "landcover.ai.v1.zip"
    DATA_DIR = "./data"
    RAW_DIR = "./data/raw"
    PROCESSED_DIR = "./data/processed"

    # Model parameters
    INPUT_CHANNELS = 3  # RGB
    NUM_CLASSES = 5  # Background + 4 classes

    # Loss parameters
    LOSS_SMOOTH = 1e-6

    # Training parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    WEIGHT_DECAY = 1e-4

    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Processing parameters
    PATCH_SIZE = 512  # Patch size
    SLICING_STRIDE = 256
