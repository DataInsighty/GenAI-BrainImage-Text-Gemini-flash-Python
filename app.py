import streamlit as st
import google.generativeai as genai
from pathlib import Path
from api_key import api_key


# configure gen Ai
genai.configure(api_key=api_key)

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}


#apply safety settings
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

## Function to load Google Gemini Pro Vision API And get response
model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                            generation_config=generation_config,
                            safety_settings=safety_settings
                            )

##initialize our streamlit app

st.set_page_config(page_title="Brain Image Analytics", page_icon=":robot")

st.image("logo.png", width=200)

st.title("Brain Image Analytics")

st.subheader("An application that can help users identify Brain Tumour")

# File uploader to upload image
uploaded_file = st.file_uploader("Choose your brain image for Analysis...", type=["jpg", "jpeg", "png", "tiff"])

# If an image is uploaded, visualize it
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Brain Image", use_column_width=True)

submit = st.button("Generate the Analysis")

## If submit button is clicked
if submit and uploaded_file:
    image_data = uploaded_file.getvalue()

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

    #generate a response based on the prompt image
    response = model.generate_content(prompt_parts)
    st.write(response.text)
