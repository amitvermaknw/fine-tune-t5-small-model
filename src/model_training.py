import torch
from tokenizer import Tokenizer 
import evaluate
from transformers import TrainingArguments, Trainer

class ModelTraining:
    def __init__(self):
        self.metric = evaluate.load("rouge")
        self.ds = Tokenizer().processing_dataset()

    def calculate_metrics(self, pre_evaluation):
        prediction, labels = pre_evaluation
        decode_prediction = self.ds.ct.tokenizer.batch_decode(prediction, skip_special_token=True)
        labels =  torch.where(labels != -100, labels, self.ds.ct.tokenizer.pad_token_id) #Replacing -100 in the lables as we can't decode them
        decode_labels = self.ds.ct.tokenizer.batch_decode(labels, skip_special_tokens=True)

        #Rouge expect a list of string for pred and ref
        result = self.metric.compute(predictions=decode_prediction, references=decode_labels, use_stemmer=True)
        result = {k:round(v*100, 4) for k,v in result.items()}
        return result
    

    def init_training(self):
        model_training_class = ModelTraining()
        training_argument = TrainingArguments(
            output_dir= "./models/t5_summarization",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=500,
            save_strategy="epoch",
            load_best_model_at_end=True, # Load the best model based on evaluations
            metric_for_best_model="rouge1",
            fp16=torch.cuda.is_available(),
            learning_rate=2e-5,
            eval_strategy="epoch"
        )

        trainer = Trainer(
            model=self.ds.ct.model,
            args=training_argument,
            train_dataset= self.ds.tokenized_train_dataset,
            eval_dataset=self.ds.tokenized_validation_dataset,
            compute_metrics=model_training_class.calculate_metrics,
            tokenizer=self.ds.ct.tokenizer
        )

        trainer.train()

        return {self.ds.ct.model, self.ds.ct.tokenizer}
        