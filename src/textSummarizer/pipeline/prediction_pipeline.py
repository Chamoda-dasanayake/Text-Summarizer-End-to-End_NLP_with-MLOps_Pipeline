from src.textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)

    def predict(self,text):
        # Truncate text before tokenization to save massive RAM spikes
        text = text[:5000] 
        
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

        print("Dialogue:")
        print(text[:200] + "..." if len(text) > 200 else text)

        inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = self.model.generate(inputs["input_ids"], **gen_kwargs)
        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        print("\nModel Summary:")
        print(output)

        return output