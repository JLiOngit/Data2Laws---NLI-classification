import argparse
from .config import FineTuning_Config as cfg
from .data_utils import *
from .classifier import NLI_FineTuner
from .label_descriptions import *


NLI_MODEL = {'deberta_large': 'MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33',
             'mdeberta_base': 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli',
             'roberta_large': 'roberta-large-mnli'}
LABEL_NAME = ['group', 'label']
DESCRIPTION_MAP = {"group": IMPROVED_GROUP_DESCRIPTIONS,
                    "label": LABEL_DESCRIPTIONS}

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune NLI model")
    parser.add_argument('--model', type=str, choices=NLI_MODEL.keys(), default='deberta_large', help="Select NLI model")
    parser.add_argument('--label_name', type=str, choices=LABEL_NAME, default='group', help="Name of the label column")
    parser.add_argument('--label_verbalization', type=bool, default=True, help="Use label verbalization")
    parser.add_argument('--train_path', type=str, default=cfg.TRAIN_PATH, help="Path to training data")
    parser.add_argument('--test_path', type=str, default=cfg.TEST_PATH, help="Path to test data")
    return parser.parse_args()

def main():
    args = parse_args()
    # loaded the datasets
    train_df, test_df = load_data(args.train_path, args.test_path)
    # initialized the finetuner
    nli_finetuner = NLI_FineTuner(
        model_name=NLI_MODEL[args.model],
        label_name=args.label_name,
        label_verbalization=args.label_verbalization,
        description_map=DESCRIPTION_MAP,
        config=cfg
    )
    # preprocessed and split datasets
    dataset = nli_finetuner.preprocess(train_df, test_df)
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']
    # finetuned the model
    nli_finetuner.run(train_dataset, validation_dataset)
    # tested the model
    nli_finetuner.predict(test_dataset, "./test_results")


if __name__ == "__main__":
    main()

