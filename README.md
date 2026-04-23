# VeriNews
A Streamlit app that uses a fine-tuned RoBERTa model to classify news articles as real or fake, with live reference articles fetched from the web for verification.

Model used: roberta-base, fine-tuned on LIAR, ISOT, WelFake Dataset
# Run Locally
pip install -r requirements.txt

streamlit run app.py

