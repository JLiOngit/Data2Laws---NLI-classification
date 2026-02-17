from dataclasses import dataclass


@dataclass
class FineTuning_Config:
    # split the dataset into train/validation
    SAMPLE_THRESHOLD : int = 800
    TRAINING_RATIO : float = 0.9
    RANDOM_STATE : int = 0
    LABEL_NAME : str = 'group'
    # NLI Labels mapping
    ID2LABEL = {0: "CONTRADICTION", 1: "ENTAILMENT"}
    LABEL2ID = {"CONTRADICTION":0, "ENTAILMENT": 1}
    # Lora config
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    # Training config
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-5
    WEIGHT_RATIO = 0.1
    WEIGHT_DECAY = 0.01