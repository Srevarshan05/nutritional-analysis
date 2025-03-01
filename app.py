import streamlit as st
from groq import Groq
from paddleocr import PaddleOCR
import csv

# Initialize OCR and Groq
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # English OCR support
client = Groq(api_key="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")

# Global variables
csv_path = 'data.csv'  # Not used for medical data now
log_csv_path = 'refined_text_log.csv'  # CSV file for logging refined text

# Custom CSS for mobile-friendly and professional UI
st.markdown("""
    <style>
    .main {
        background-color: #e0f7e9;
        padding: 10px;
    }
    .title {
        font-size: 24px;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
    }
    .stButton>button {
        background-color: #66bb6a;
        color: white;
        border-radius: 8px;
        padding: 10px;
        width: 100%;
    }
    .stTextArea textarea {
        font-size: 16px;
        border: 2px solid #2e7d32;
        border-radius: 8px;
    }
    .stSelectbox {
        width: 100%;
    }
    .image-container {
        border: 2px dashed #2e7d32;
        padding: 10px;
        text-align: center;
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

def log_refined_text(refined_text):
    """Logs the refined OCR-extracted text into a CSV file."""
    with open(log_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([refined_text])

def extract_and_clean_text(image_path):
    """Extracts text from an image using OCR and cleans it."""
    results = ocr.ocr(image_path, cls=True)
    cleaned_text = []
    for line in results[0]:
        text = line[1][0]
        if len(text) >= 3 and text.isprintable():
            cleaned_text.append(text)
    return cleaned_text

def process_image_and_csv(image_path, csv_path):
    """Processes a nutritional image, sending data to the Groq API for analysis."""
    if not image_path:
        return "Error: No image path provided!", "", ""
    try:
        # Extract text from the image
        cleaned_text = extract_and_clean_text(image_path)
        extracted_text = ", ".join(cleaned_text)

        # Sending the OCR text to NLP model
        prompt = f"""
        I am using OCR to extract the Nutritional information from the Food pack labels. 
        I need you to refine the text: {extracted_text}. Just return the nutritional facts and Ingredients. 
        Based on the ingredients, what is the Food name? No other words.
        """
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "You are a professional medical advisor."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
            top_p=1,
            stream=False
        )
        refined_text = response.choices[0].message.content
        log_refined_text(refined_text)
        return refined_text, "", ""
    except Exception as e:
        return f"Error occurred: {e}", "", ""

def process_medical_report(image_path):
    """Processes a medical report image and refines it with Groq."""
    extracted_text = extract_and_clean_text(image_path)
    medical_prompt = f"""
    I am using OCR to extract the text from a medical report. 
    Refine the text: {extracted_text}, remove any noise, and provide a clear summary of the medical findings. 
    Just return the important medical details and diagnosis if available, no extra words.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are a professional medical advisor."},
                  {"role": "user", "content": medical_prompt}],
        temperature=0.7,
        max_tokens=300,
        top_p=1,
        stream=False
    )
    return response.choices[0].message.content

def evaluate_combined(nutritional_text, medical_text, model, language):
    """Evaluates combined nutritional and medical data."""
    next_prompt = f"""
    Dear User,

    Based on the extracted text from your food pack labels: {nutritional_text},
    and the details from your medical report: {medical_text},
    please evaluate the ingredients for safety.
    Provide a short recommendation on whether the food is safe to consume,
    including the safe quantity for intake if applicable.
    If the food is not recommended, briefly explain why it should be avoided.

    Please provide the response in the following format:

    1. First, a short and clear recommendation in **English**.
    2. After that, a short and clear recommendation in **{language}** that corresponds to the English response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a professional medical advisor."},
                  {"role": "user", "content": next_prompt}],
        temperature=0.7,
        max_tokens=400,
        top_p=1,
        stream=False
    )
    return response.choices[0].message.content

def main():
    st.title("Nutritional & Medical Analysis Tool", anchor="title")

    # Nutritional Section
    st.markdown("<div class='section-title'>Nutritional Data</div>", unsafe_allow_html=True)
    nutritional_image = st.file_uploader("Upload Nutritional Image", type=["jpg", "png", "jpeg"])
    if nutritional_image:
        st.image(nutritional_image, caption="Uploaded Nutritional Image", use_container_width=True, clamp=True, output_format="auto")  # Updated parameter
        image_path = nutritional_image.read()  # Read bytes for OCR
        if st.button("Analyze Nutritional Data"):
            refined_text, _, _ = process_image_and_csv(image_path, csv_path)
            st.session_state['nutritional_text'] = refined_text
            st.text_area("Nutritional Analysis Output", value=refined_text, height=200, key="nutritional_output")

    # Medical Report Section
    st.markdown("<div class='section-title'>Medical Report</div>", unsafe_allow_html=True)
    medical_image = st.file_uploader("Upload Medical Report", type=["jpg", "png", "jpeg"])
    if medical_image:
        st.image(medical_image, caption="Uploaded Medical Report", use_container_width=True, clamp=True, output_format="auto")  # Updated parameter
        if st.button("Process Medical Report"):
            medical_refined_text = process_medical_report(medical_image.read())
            st.session_state['medical_text'] = medical_refined_text
            st.text_area("Medical Report Analysis Output", value=medical_refined_text, height=200, key="medical_output")

    # Combined Evaluation Section
    st.markdown("<div class='section-title'>Combined Evaluation</div>", unsafe_allow_html=True)
    model = st.selectbox("Select Model", [
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "llama-3.2-1b-preview",
        "qwen-2.5-32b"
    ])
    language = st.selectbox("Select Language", ["English", "Tamil", "Hindi"])
    
    if st.button("Evaluate Combined Data"):
        nutritional_text = st.session_state.get('nutritional_text', '')
        medical_text = st.session_state.get('medical_text', '')
        if not nutritional_text or not medical_text:
            st.error("Please analyze both nutritional data and medical report first.")
        else:
            combined_output = evaluate_combined(nutritional_text, medical_text, model, language)
            st.text_area("Combined Evaluation Output", value=combined_output, height=300, key="combined_output")

if __name__ == "__main__":
    main()