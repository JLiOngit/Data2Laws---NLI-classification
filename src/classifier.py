import os
import torch as th
import pandas as pd
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score

from data_utils import *
from .label_descriptions import GROUP_DESCRIPTIONS, LABEL_DESCRIPTIONS
from .config import FineTuning_Config


class ZeroShotClassifier:

    def __init__(self,
                 model_name,
                 batch_size = 32,
                 multi_prediction = False,
                 template = "This GitHub comment is about {}."):
        self.model_name = model_name
        self.batch_size = batch_size
        self.multi_labels = multi_prediction
        self.template = template
        device = 0 if th.cuda.is_available() else -1
        self.classifier = pipeline('zero-shot-classification', model=model_name, device=device)

    def classification(self, test_path, label_name = "group", label_verbalization=True):
        test_df = pd.read_json(test_path, lines=True)
        test_dataset = Dataset.from_pandas(test_df)
        try:
            candidate_labels = test_df[label_name].unique()
        except KeyError:
            raise KeyError(f"The label '{label_name}' is not present in the columns of the dataframe.")
        if label_verbalization:
            DESCRIPTION_MAP = {"group": GROUP_DESCRIPTIONS,
                               "label": LABEL_DESCRIPTIONS}
            try:
                label_descriptions = DESCRIPTION_MAP[label_name]
            except KeyError:
                raise KeyError(f"The label'{label_name}' is not present in the columns of the description dataframe.")
            assert set(candidate_labels) == set(label_descriptions[label_name].unique())
            candidate_labels = label_descriptions[label_name + '_description'].unique()
        def collate_function(batch):
            messages = batch['message']
            predictions, scores = [], []
            for message in messages:
                result = self.classifier(message, candidate_labels=candidate_labels, multi_labels=self.multi_labels, hypothesis_template=self.template)
                predicted_label = result['labels'][0]
                score = result['scores'][0]
                if label_verbalization:
                    predicted_label = label_descriptions.loc[label_descriptions[label_name + '_description'] == predicted_label, label_name].iloc[0]
                predictions.append(predicted_label)
                scores.append(score)
            return {'predicted_'+ label_name: predictions, 'scores': scores}
        test_dataset = test_dataset.map(collate_function, batched=True, batch_size=self.batch_size)
        test_df = pd.DataFrame(test_dataset)
        return test_df


class SavePeftModelCallback(TrainerCallback):

    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control


class NLI_FineTuning:

    def __init__(self,
                 model_name,
                 label_name,
                 label_verbalization,
                 fine_tuning_config : FineTuning_Config):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self.lora_model(model_name)
        self.label_name = label_name
        self.label_verbalization = label_verbalization
        self.config = fine_tuning_config
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

    def lora_model(self, model_name):
        model = AutoModelForSequenceClassification(model_name,
                                                   num_labels=len(self.config.ID2LABEL),
                                                   id2label=self.config.ID2LABEL,
                                                   label2id=self.config.LABEL2ID)
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                 r = self.config.LORA_R,
                                 lora_alpha = self.config.LORA_ALPHA,
                                 lora_dropout = self.config.LORA_DROPOUT,
                                 target_modules= ['query_proj', 'value_proj'])
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def preprocess(self, train_df, test_df):
        # switch to label descriptions
        if self.label_verbalization:
            DESCRIPTION_MAP = {"group": GROUP_DESCRIPTIONS,
                               "label": LABEL_DESCRIPTIONS}
            try:
                label_descriptions = DESCRIPTION_MAP[self.label_name]
            except KeyError:
                raise KeyError(f"The label'{self.label_name}' is not present in the columns of the description dataframe.")
            train_df = pd.merge(train_df, label_descriptions, on=self.label_name, how='inner')
            test_df = pd.merge(test_df, label_descriptions, on=self.label_name, how='inner')
        # split the train dataset into train / validation
        train_df, validation_df = split_train_dataset(train_df, self.config.SAMPLE_THRESHOLD, self.config.TRAINING_RATIO, self.config.RANDOM_STATE)
        # create false labels to carry out nli classification
        nli_train_df = create_false_labels(train_df, self.label_name)
        nli_validation_df = create_false_labels(validation_df, self.label_name)
        nli_test_df = create_false_labels(test_df, self.label_name)
        # tokenize the premise (message) and the hypothesis (label or label description)
        dataset = DatasetDict({
            'train': Dataset.from_pandas(nli_train_df),
            'validation': Dataset.from_pandas(nli_validation_df),
            'test': Dataset.from_pandas(nli_test_df)
        })
        def tokenize_premise_hypothesis(row):
            premise = row['message']
            hypothesis = row[self.label_name + '_description']
            return self.tokenizer(text=premise, text_pair=hypothesis, truncation='only_first')        
        dataset = dataset.map(tokenize_premise_hypothesis, batched=True, batch_size=self.config.BATCH_SIZE)
        return dataset

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        balanced_accuracy = balanced_accuracy_score(labels, preds)
        return {"f1": f1, "precision": precision, "recall": recall, "balanced accuracy":balanced_accuracy}

    def fine_tune(self, train_dataset, validation_dataset):
        # define training arguments
        train_arguments = TrainingArguments(
            output_dir = './train_results',
            num_train_epochs = self.config.NUM_EPOCHS,
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = self.config.BATCH_SIZE // 8,
            per_device_eval_batch_size = 8,
            learning_rate = self.config.LEARNING_RATE,
            weight_ratio = self.config.WEIGHT_RATIO,
            weight_decay = self.config.WEIGHT_DECAY,
            logging_steps = 100,
            eval_strategy = 'steps',
            eval_steps = 300,
            save_strategy = 'steps',
            save_steps = 300,
            metric_for_best_model = 'f1',
            load_best_model_at_end = True,
            fp16 = th.cuda.is_available()
        )
        # define the trainer
        trainer = Trainer(
            model = self.model,
            args = train_arguments,
            train_dataset = train_dataset,
            eval_dataset = validation_dataset,
            data_collator = self.data_collator,
            compute_metrics = self.compute_metrics,
            callbacks=[SavePeftModelCallback()]
        )
        trainer.train()
        validation_results = trainer.evaluate()
        return validation_results
    
    def test(self, test_dataset):
        # define the trainer
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="./test_results", per_device_eval_batch_size=16),
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
        metrics = trainer.evaluate(test_dataset)
        metrics = {k.replace('eval_', ''): v for k, v in metrics.items()}
        # save metrics
        with open(f"test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        # print metrics
        for key, value in metrics.items():
            print(f"{key.capitalize()}: {value:.4f}")


    
    




    


    


    
