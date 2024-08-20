import regex as re
from langdetect import detect
from nltk.corpus import stopwords

from io import StringIO
import nltk
import pandas as pd
import numpy as np
import torch
import random
from transformers import DistilBertModel, DistilBertTokenizer, BertModel, BertTokenizer
import joblib
import streamlit as st

nltk.download('stopwords')

# set seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

stop = stopwords.words('english')


def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in (stop)])
    return cleaned_text

def get_embeddings(sample_comments, model, tokenizer):
    # Clean the text
    sample_comments = [clean_text(comment) for comment in sample_comments]

    # Tokenize and pad in one step
    encoded_input = tokenizer(sample_comments, 
                              padding=True, 
                              truncation=True, 
                              return_tensors='pt',
                              max_length=1024)  # Adjust max_length as needed

    # Move to the same device as the model
    input_ids = encoded_input['input_ids'].to(model.device)
    attention_mask = encoded_input['attention_mask'].to(model.device)

    # Get the embeddings
    with torch.no_grad():
        model_output = model(input_ids, attention_mask=attention_mask)
    
    # Extract the [CLS] token embeddings
    embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()

    return embeddings

def pipeline(comments, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    embeddings = get_embeddings(comments, model, tokenizer)
    model = joblib.load(model_path)
    predictions = model.predict(embeddings)
    sentiment_map = {-1: 'Negative', 1: 'Positive', 0: 'Neutral'}
    predictions = [sentiment_map[pred] for pred in predictions]
    prediction_df = pd.DataFrame({'comment': comments, 'sentiment': predictions})
    return prediction_df

def process_comment(comment):
    result = pipeline([comment], model_path='sentiment_analysis_model.pkl')
    if result['sentiment'][0] == 'Positive':
        return "5 stars"
    elif result['sentiment'][0] == 'Negative':
        return "1 star"
    else:
        return "3 stars"

def main():
    
    st.set_page_config("Movie Review Analysis")
    
    st.header("Know Your Audience From 🍅 to Reddit.")
    st.write("With the help of BERT LLM, you can predict the sentiment of movie reviews+. By collecting comments from Rotten Tomatoes Audiences, we can see how people react to the movies and predict the sentiment on Reddit comments.")
    st.write("In the future, we will upgrade this website to make you throw in a Reddit link and use it to predict the score for this movie from audiences!")

    
    # user_input = st.text_input("Test the comment:","You could HEAR the disappointment from the audience I was with when that â€œTo Be Continued title card showed up lmao. But besides that, yeah this was a dang fine sequel. Better than most I dare say. Seeing aged-up Peni Parker in her upgraded mech for the first time almost made me tear up a bit.Oh yeah and Hailee Steinfeld deserves an Oscar for her performance here, all of the scenes when she's paired up with her dad were beautifully executed. Easily the best parts of the film for me personally.")
    user_input = st.text_area(
    "Test the comment:",
    placeholder="You could HEAR the disappointment from the audience I was with when that â€œTo Be Continued title card showed up lmao. But besides that, yeah this was a dang fine sequel. Better than most I dare say. Seeing aged-up Peni Parker in her upgraded mech for the first time almost made me tear up a bit.Oh yeah and Hailee Steinfeld deserves an Oscar for her performance here, all of the scenes when she's paired up with her dad were beautifully executed. Easily the best parts of the film for me personally.",
    height=200
)
    user_input = [user_input]

    if st.button("Predict the sentence"):
        result = pipeline(user_input, model_path='sentiment_analysis_model.pkl')
        if result['sentiment'][0] == 'Positive':
            answer = "5 stars"
        elif result['sentiment'][0] == 'Negative':
            answer = "1 star"
        else:
            answer = "3 stars"
        st.success(f"The score of the comment is: {answer}")

    # uploaded_file = st.file_uploader("Or upload a CSV file with comments", type="csv")
    # if st.button("Predict the CSV File") and uploaded_file:
    #     df = pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")))
    #     if 'comment' not in df.columns:
    #         st.error("The CSV file must have a 'comment' column.")
    #     else:
    #         # Process each comment in the CSV
    #         df['comment'] = df['comment'].astype(str).apply(pipeline, model_path='sentiment_analysis_model.pkl')
    #         df['score'] = df['comment'].apply(process_comment)
            
    #         # Display the results
    #         st.write("Results from CSV:")
    #         st.dataframe(df)
            
    #         # Option to download results
    #         csv = df.to_csv(index=False)
    #         st.download_button(
    #             label="Download results as CSV",
    #             data=csv,
    #             file_name="sentiment_analysis_results.csv",
    #             mime="text/csv",
    #         )


if __name__ == "__main__":
    main()
