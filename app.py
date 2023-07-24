import streamlit as st
import tensorflow as tf
#from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Load the pre-trained BERT model and tokenizer
# Experiment:1
#{
#model_name = 'bert-base-cased'
#tokenizer = BertTokenizer.from_pretrained(model_name)
#model = TFBertForSequenceClassification.from_pretrained(model_name)
#}
# Experiement:2
#{ Fine-tuned model size execeds to allowed limit
# Load tokenizer
#tokenizer = AutoTokenizer.from_pretrained("./Models/sentiment_learning_tensorflow/")
# Load model
#model = TFAutoModelForSequenceClassification.from_pretrained('./Models/sentiment_learning_tensorflow/')
#}

#Experiment:3
#model_name = 'distilbert-base-uncased'

#Experiment:4
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

# Tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load model
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def predict_sentiment(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        #max_length=128,
        max_length=512,
        return_tensors='tf',
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
    )
    
    # Prediction
    outputs = model(inputs)
    logits = outputs.logits
    prediction = tf.argmax(logits, axis=1).numpy()[0]    
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment

def main():
    st.title("Sentiment Analysis App 1.1")

    user_input = st.text_area("Enter your text here:")
    if st.button("Analyze"):
        if user_input.strip() != "":
            sentiment = predict_sentiment(user_input)
            st.write(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
