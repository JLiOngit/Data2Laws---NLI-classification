import torch as th
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from peft import LoraConfig, get_peft_model


class ZeroShotClassifier:

    def __init__(self,
                 model_name,
                 batch_size=32,
                 multi_labels=False,
                 template="This GitHub comment is about {}."):
        self.model_name = model_name
        self.batch_size = batch_size
        self.multi_labels = multi_labels
        self.template = template
        device = 0 if th.cuda.is_available() else -1
        self.classifier = pipeline('zero-shot-classification', model=model_name, device=device)

    def classification(self, df):
        dataset = Dataset.from_pandas(df)
        candidate_labels = df['labels'].unique()
        def collate_function(batch):
            messages = batch['message']
            predictions, scores = [], []
            for message in messages:
                result = self.classifier(message, candidate_labels=candidate_labels, multi_labels=self.multi_labels, hypothesis_template=self.template)
                predictions.append(result['labels'])
                scores.append(result['scores'])
            return {'predicted_label': predictions, 'scores': scores}
        dataset = dataset.map(collate_function, batched=True, batch_size=self.batch_size)
        return dataset

