import torch
from transformers import AutoTokenizer, BertForSequenceClassification

model_path = "F:\\code\\self\\bert_class\\results\\checkpoint-381"
model_path_tokenizer="F:\\code\\self\\bert_class\\hgmodel\\bert_base_chinese"
tokenizer = AutoTokenizer.from_pretrained(model_path_tokenizer)
model = BertForSequenceClassification.from_pretrained(model_path)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]