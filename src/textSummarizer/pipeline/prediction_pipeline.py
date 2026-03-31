from src.textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self,text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
        
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

        print("Dialogue:")
        print(text)

        inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = model.generate(inputs["input_ids"], **gen_kwargs)
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        print("\nModel Summary:")
        print(output)

        return output