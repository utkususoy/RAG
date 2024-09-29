from transformers import AutoTokenizer, AutoModelForCausalLM

class CustomLLM:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_length: int = 512):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output_sequences = self.model.generate(
            inputs['input_ids'], max_length=max_length, do_sample=True
        )
        return self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# # Example usage
# llm = CustomLLM("gpt2")
# response = llm.generate("What is LangChain?")
# print(response)
