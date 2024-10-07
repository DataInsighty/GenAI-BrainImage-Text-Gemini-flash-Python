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
As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining medical images for a renowned hospital. Your expertise is crucial in identifying any anomalies, diseases, or health issues that may be present in the images.

Your Responsibilities include:
1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal findings.
2. Findings Report: Document all observed anomalies or signs of disease. Clearly articulate these findings in a structured format.
3. Recommendations and Next Steps: Based on your analysis, suggest potential next steps, including further tests or treatments as applicable.
4. Treatment Suggestions: If appropriate, recommend possible treatment options or interventions.

Important Notes:
1. Scope of Response: Only respond if the image pertains to human health issues.
2. Clarity of Image: In cases where the image quality impedes clear analysis, note that certain aspects are "Unable to be determined based on the provided image."
3. Disclaimer: Accompany your analysis with the disclaimer: “This analysis is intended as a preliminary review. Consult with a licensed medical doctor before making any diagnostic or treatment decisions. Further tests may be required to confirm these findings.”
4. Your insights are invaluable in guiding clinical decisions. Please proceed with the analysis, adhering to the structured approach outlined above.

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
