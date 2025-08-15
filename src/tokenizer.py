from transformers import T5ForConditionalGeneration, T5Tokenizer
from const import Constant 
from load_dataset import LoadDataset

class Tokenizer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.tokenized_train_dataset = None
        self.tokenized_validation_dataset =None

    def create_token(self, model_name:str):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        return self

    def add_prefix(self, articles):
        inputs = [f"summarize: {document}" for document in articles["article"]]
        model_token_input = self.tokenizer(
            inputs, 
            max_length=Constant.MAX_INPUT_LEN, 
            truncation=True,
            padding="max_length",
            return_tensors="pt")
         
        labels = self.tokenizer(
            text_target=articles["highlights"], 
            max_length=Constant.MAX_OUTPUT_LEN, 
            truncation=True,
            padding="max_length",
            return_tensors="pt")
        
        model_token_input["labels"] = labels["input_ids"]
        return model_token_input
    
    def processing_dataset(self):
        tokenizer_class = Tokenizer()
        self.ct = tokenizer_class.create_token(Constant.MODEL_NAME)

        print("Preprocessing the dataset")
        dataset = LoadDataset().load_data()
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        self.tokenized_train_dataset = train_dataset.map(tokenizer_class.add_prefix, batched=True, remove_columns=["article", "highlights", "id"])
        self.tokenized_validation_dataset =validation_dataset.map(tokenizer_class.add_prefix, batched=True, remove_columns=["article", "highlights", "id"])
        print("Dataset processing completed")

        return self