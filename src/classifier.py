import torch as th
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from .label_descriptions import GROUP_DESCRIPTIONS, LABEL_DESCRIPTIONS


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

    def classification(self, df, label_name = "group", label_verbalization=True):
        dataset = Dataset.from_pandas(df)
        candidate_labels = df[label_name].unique()
        if label_verbalization:
            DESCRIPTION_MAP = {"group": GROUP_DESCRIPTIONS,
                               "label": LABEL_DESCRIPTIONS}
            try:
                df_descriptions = DESCRIPTION_MAP[label_name]
            except KeyError:
                raise KeyError(f"Label name '{label_name}' is not handled.")
            assert set(candidate_labels) == set(df_descriptions[label_name].unique())
            candidate_labels = df_descriptions[label_name + '_description'].unique()

        def collate_function(batch):
            messages = batch['message']
            predictions, scores = [], []
            for message in messages:
                result = self.classifier(message, candidate_labels=candidate_labels, multi_labels=self.multi_labels, hypothesis_template=self.template)
                predicted_label = result['labels'][0]
                score = result['scores'][0]
                if label_verbalization:
                    predicted_label = df_descriptions.loc[df_descriptions[label_name + '_description'] == predicted_label, label_name].iloc[0]
                predictions.append(predicted_label)
                scores.append(score)
            return {'predicted_'+ label_name: predictions, 'scores': scores}
        
        dataset = dataset.map(collate_function, batched=True, batch_size=self.batch_size)
        final_df = pd.DataFrame(dataset)
        return final_df


