import streamlit as st
import google.generativeai as genai
from pathlib import Path
from io import BytesIO
from PIL import Image
from api_key import api_key

# Configure Gen AI
genai.configure(api_key=api_key)

# Adjusted generation configuration
generation_config = {
    "temperature": 0.7,  # Lower temperature for more focused output
    "top_p": 0.9,        # Reduced top_p to limit response length
    "top_k": 50,         # Limit the number of choices to consider
    "max_output_tokens": 2048,  # Reduce max tokens to control output length
    "response_mime_type": "text/plain",
}

# Apply safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

system_prompt = """
You are a highly skilled medical expert specializing in the analysis of brain images for medical analytics purposes, tasked with identifying any anomalies or irregularities. Your primary objective is to carefully review the provided brain scan and:
1. Identify potential patterns, general structures, or any areas of interest based on the image provided.
2. Identify potential signs of abnormalities, such as tumors, lesions, or structural issues.
2. Provide a detailed findings report, describing the observed issues clearly and concisely.
3. Recommend potential follow-up steps, such as further diagnostic tests, imaging, or biopsies.
4. Suggest relevant treatment options or interventions based on the identified issues, if applicable.

Important notes:
- Only respond if the image pertains to human health issues.
- If the image quality is poor, note any areas where analysis is inconclusive.
- If no abnormalities are found, include a statement indicating that no anomalies were detected.
- Include the following disclaimer in your analysis: "This analysis is intended as a preliminary review. Consult with a licensed medical doctor before making any diagnostic or treatment decisions. Further tests may be required to confirm these findings."
"""



# Function to load Google Gemini Pro Vision API And get response
model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Initialize our Streamlit app
st.set_page_config(page_title="Brain Image Analytics", page_icon=":robot")

st.image("logo.png", width=200)

st.title("Brain Image Analytics")
st.subheader("An application that helps users to identify general visual patterns in brain images for educational purposes.")

# File uploader for brain image
uploaded_file = st.file_uploader("Choose your brain image for analysis...", type=["jpg", "jpeg", "png", "tiff"])

submit = st.button("Generate the Analysis")

# If submit button is clicked and file is uploaded
if uploaded_file:
    # Resize the uploaded image to avoid large input context
    img = Image.open(uploaded_file)
    img = img.resize((512, 512))  # Resize to 512x512
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    image_data = buffer.getvalue()

    # Display the resized image in the app
    st.image(img, caption="Uploaded Brain Image (Resized)", use_column_width=True)

    if submit:
        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": image_data
            }
        ]

        prompt_parts = [
            image_parts[0],
            system_prompt,
        ]

        # Generate a response based on the prompt image
        try:
            response = model.generate_content(prompt_parts)
            st.write(response.text)
        except Exception as e:
            st.error(f"An error occurred: {e}")
