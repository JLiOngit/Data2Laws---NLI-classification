import os
import torch as th
import numpy as np
import random
import pandas as pd
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
from .data_utils import *


def get_description_df(label_name, description_map):
    try:
        label_descriptions = description_map[label_name]
    except KeyError:
        raise KeyError(f"The label'{label_name}' is not present in the columns of the description dataframe.")
    return label_descriptions


class ZeroShotClassifier:

    def __init__(self,
                 model_name,
                 label_name,
                 label_verbalization,
                 description_map,
                 batch_size = 32,
                 multi_prediction = False,
                 template = "This GitHub comment is about {}."):
        self.model_name = model_name
        self.label_name = label_name
        self.label_verbalization = label_verbalization
        self.description_map = description_map
        self.batch_size = batch_size
        self.multi_labels = multi_prediction
        self.template = template
        device = 0 if th.cuda.is_available() else -1
        self.classifier = pipeline('zero-shot-classification', model=model_name, device=device)

    def classification(self, test_path):
        test_df = pd.read_json(test_path, lines=True)
        test_dataset = Dataset.from_pandas(test_df)
        try:
            candidate_labels = test_df[self.label_name].unique()
        except KeyError:
            raise KeyError(f"The label '{self.label_name}' is not present in the columns of the dataframe.")
        if self.label_verbalization:
            label_descriptions = get_description_df(self.label_name, self.description_map)
            assert set(candidate_labels) == set(label_descriptions[self.label_name].unique())
            candidate_labels = label_descriptions[self.label_name + '_description'].unique()
        def collate_function(batch):
            messages = batch['message']
            predictions, scores = [], []
            for message in messages:
                result = self.classifier(message, candidate_labels=candidate_labels, multi_labels=self.multi_labels, hypothesis_template=self.template)
                predicted_label = result['labels'][0]
                score = result['scores'][0]
                if self.label_verbalization:
                    predicted_label = label_descriptions.loc[label_descriptions[self.label_name + '_description'] == predicted_label, self.label_name].iloc[0]
                predictions.append(predicted_label)
                scores.append(score)
            return {'predicted_'+ self.label_name: predictions, 'scores': scores}
        test_dataset = test_dataset.map(collate_function, batched=True, batch_size=self.batch_size)
        test_df = pd.DataFrame(test_dataset)
        return test_df


class SavePeftModelCallback(TrainerCallback):

    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control


class WeightedTrainer(Trainer):

    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        class_weights = th.tensor([1.0, float(self.num_labels)], device=logits.device, dtype=logits.dtype)
        loss_fct = th.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class NLI_FineTuner:

    def __init__(self,
                 model_name,
                 label_name,
                 label_verbalization,
                 description_map,
                 config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self.lora_model(model_name)
        self.label_name = label_name
        self.label_verbalization = label_verbalization
        self.description_map = description_map
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

    def lora_model(self, model_name):
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
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
    
    def create_false_labels(self, df, num_negative_samples = None):
        colummns = ['message', self.label_name, self.label_name + '_description'] if self.label_verbalization else  ['message', self.label_name]
        positive = df[colummns].copy()
        positive['labels'] = 1
        negative_pairs = []
        label_unique = df[self.label_name].unique()
        for i in range(positive.shape[0]):
            message = positive.iloc[i]['message']
            true_class = positive.iloc[i][self.label_name]
            false_classes = [c for c in label_unique if c != true_class]
            if num_negative_samples:
                false_classes = random.sample(false_classes, min(num_negative_samples, len(false_classes)))
            for false_class in false_classes:
                negative_label = {'message': message,
                                  self.label_name: false_class,
                                  'labels': 0}
                if self.label_verbalization:
                    label_descriptions = get_description_df(self.label_name, self.description_map)
                    negative_label[self.label_name + '_description'] = label_descriptions.loc[label_descriptions[self.label_name] == false_class,self.label_name + '_description'].iloc[0]
                negative_pairs.append(negative_label)
        negative = pd.DataFrame(negative_pairs)
        final_df = pd.concat([positive, negative]).sample(frac=1).reset_index(drop=True)
        return final_df

    def preprocess(self, train_df, test_df):
        # switch to label descriptions
        if self.label_verbalization:
            label_descriptions = get_description_df(self.label_name, self.description_map)
            train_df = pd.merge(train_df, label_descriptions, on=self.label_name, how='inner')
            test_df = pd.merge(test_df, label_descriptions, on=self.label_name, how='inner')
        # split the train dataset into train / validation
        train_df, validation_df = split_train_dataset(train_df, self.config.SAMPLE_THRESHOLD, self.config.TRAINING_RATIO, self.config.RANDOM_STATE)
        # create false labels to carry out nli classification
        nli_train_df = self.create_false_labels(train_df)
        nli_validation_df = self.create_false_labels(validation_df)
        nli_test_df = self.create_false_labels(test_df)
        print(nli_test_df.head())
        # tokenize the premise (message) and the hypothesis (label or label description)
        dataset = DatasetDict({
            'train': Dataset.from_pandas(nli_train_df),
            'validation': Dataset.from_pandas(nli_validation_df),
            'test': Dataset.from_pandas(nli_test_df)
        })
        def tokenize_premise_hypothesis(batch):
            premise = list(batch['message'])
            hypothesis = list(batch[self.label_name + '_description']) if self.label_verbalization else list(batch[self.label_name])
            return self.tokenizer(text=premise, text_pair=hypothesis, truncation='only_first')        
        dataset = dataset.map(tokenize_premise_hypothesis, batched=True, batch_size=self.config.BATCH_SIZE)
        return dataset

    def compute_metrics(self, output):
        labels = output.label_ids
        preds = output.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        balanced_accuracy = balanced_accuracy_score(labels, preds)
        return {"f1": f1,
                "precision": precision,
                "recall": recall,
                "balanced accuracy":balanced_accuracy}

    def run(self, train_dataset, validation_dataset):
        # define training arguments
        train_arguments = TrainingArguments(
            output_dir = './train_results',
            num_train_epochs = self.config.NUM_EPOCHS,
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = self.config.BATCH_SIZE // 8,
            per_device_eval_batch_size = 8,
            weight_decay = self.config.WEIGHT_DECAY,
            logging_steps = 100,
            eval_strategy = 'steps',
            eval_steps = 250,
            save_strategy = 'steps',
            save_steps = 250,
            save_total_limit=1,
            metric_for_best_model = 'f1',
            load_best_model_at_end = True,
            fp16 = False
        )
        num_labels = len(np.unique(train_dataset[self.label_name]))
        trainer = WeightedTrainer(
            num_labels = num_labels,
            model = self.model,
            args = train_arguments,
            train_dataset = train_dataset,
            eval_dataset = validation_dataset,
            data_collator = self.data_collator,
            compute_metrics = self.compute_metrics,
            callbacks=[SavePeftModelCallback()]
        )

        def get_checkpoint(path):
            if not os.path.exists(path):
                return None
            checkpoints = [d for d in os.listdir(path) if d.startswith("checkpoint-")]
            if not checkpoints:
                return None
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            return os.path.join(path, checkpoints[-1])
        
        checkpoint = get_checkpoint(train_arguments.output_dir)
        trainer.train(resume_from_checkpoint=checkpoint)
        return trainer.evaluate()
    
    def compute_nli_metrics(self, test_dataset, output_dir):
        test_df = pd.DataFrame(test_dataset)
        test_df = test_df.query('labels == 1  | predicted_labels == 1')
        # created TP, FP, FN columns
        test_df['TP'] = ((test_df['labels'] == 1) & (test_df['predicted_labels'] == 1)).astype(int)
        test_df['FP'] = ((test_df['labels'] == 0) & (test_df['predicted_labels'] == 1)).astype(int)
        test_df['FN'] = ((test_df['labels'] == 1) & (test_df['predicted_labels'] == 0)).astype(int)
        # aggregated per label
        aggregated_df = test_df.groupby(self.label_name)[['TP', 'FP', 'FN']].sum()
        # calculted metrics for each label
        aggregated_df['precision'] = aggregated_df['TP'] / (aggregated_df['TP'] + aggregated_df['FP']).replace(0, pd.NA)
        aggregated_df['recall'] = aggregated_df['TP'] / (aggregated_df['TP'] + aggregated_df['FN']).replace(0, pd.NA)
        aggregated_df['f1'] = 2 * aggregated_df['precision'] * aggregated_df['recall'] / (aggregated_df['precision'] + aggregated_df['recall'])
        aggregated_df = aggregated_df.fillna(0)
        # displayed and stored the results
        results = {}
        print("Class Scores")
        for label, row in aggregated_df.iterrows():
            results[label] = {"precision": float(row['precision']),
                                            "recall": float(row['recall']),
                                            "f1": float(row['f1'])}
            print(f"Class: {label}")
            print(f"Precision: {row['precision']:.4f}")
            print(f"Recall: {row['recall']:.4f}")
            print(f"F1-score: {row['f1']:.4f}")
            print('-' * 30)
        precision_macro = aggregated_df['precision'].mean()
        recall_macro = aggregated_df['recall'].mean()
        f1_macro = aggregated_df['f1'].mean()
        results["macro"] = {"precision": float(precision_macro),
                            "recall": float(recall_macro),
                            "f1": float(f1_macro)}
        print("\nMacro Scores")
        print(f"Precision: {results["macro"]['precision']:.4f}")
        print(f"Recall: {results["macro"]['recall']:.4f}")
        print(f"F1-score: {results["macro"]['f1']:.4f}")
        # saved into a json file
        with open(os.path.join(output_dir, f"{self.label_name}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        return results

    
    def predict(self, test_dataset, output_dir):
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir=output_dir, per_device_eval_batch_size=16),
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
        outputs = trainer.predict(test_dataset)
        prediction = np.argmax(outputs.predictions, axis=-1)
        test_dataset = test_dataset.add_column("predicted_labels", prediction)
        self.compute_nli_metrics(test_dataset, output_dir)
