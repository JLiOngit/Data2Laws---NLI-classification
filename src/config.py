from dataclasses import dataclass


@dataclass
class FineTuning_Config:
    TRAIN_PATH = "code_review_comments/crc_train.jsonl"
    TEST_PATH = "code_review_comments/crc_test.jsonl"
    # split the dataset into train/validation
    SAMPLE_THRESHOLD : int = 800
    TRAINING_RATIO : float = 0.9
    RANDOM_STATE : int = 0
    # NLI labels mapping
    ID2LABEL = {0: "CONTRADICTION", 1: "ENTAILMENT"}
    LABEL2ID = {"CONTRADICTION":0, "ENTAILMENT": 1}
    # Lora configuration
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    # Training configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01