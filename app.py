import pandas as pd
import pickle
from pypdf import PdfReader
import re
import streamlit as st

# Load models
word_vector = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText



def categorize_resumes(uploaded_files):
    results = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            page = reader.pages[0]
            text = page.extract_text()
            cleaned_resume = cleanResume(text)

            input_features = word_vector.transform([cleaned_resume])

            probabilities = model.predict_proba(input_features)[0]
            top_indices = probabilities.argsort()[-3:][::-1]
            top_categories = le.inverse_transform(top_indices)

            results.append({
                'filename': uploaded_file.name,
                'Top 1': top_categories[0],
                'Top 2': top_categories[1],
                'Top 3': top_categories[2]
            })

    results_df = pd.DataFrame(results)
    return results_df


st.title("Resume Categorizer Application")
st.subheader("With Python & Machine Learning")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if st.button("Categorize Resumes"):
    if uploaded_files:
        results_df = categorize_resumes(uploaded_files)
        st.write(results_df)
        results_csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=results_csv,
            file_name='categorized_resumes.csv',
            mime='text/csv',
        )
        st.success("Resumes categorization and processing completed.")
    else:
        st.error("Please upload files and specify the output directory.")