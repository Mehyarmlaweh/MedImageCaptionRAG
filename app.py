"Streamlit interfcae"
import requests
import streamlit as st

# FastAPI endpoint URL
FASTAPI_URL = "http://localhost:8000/caption/"

# Streamlit UI
st.title("Medical Image Captioning")
st.write("""
This application generates medical image descriptions using
a combination of retrieval-augmented generation (RAG) and traditional methods.
""")

# File upload for the image
image_file = st.file_uploader("Upload a medical image (JPG/PNG)",
                              type=["jpg", "jpeg", "png"]
                              )

if image_file:
    # Display the uploaded image
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    # Submit button
    if st.button("Generate Caption"):
        with st.spinner("Processing the image..."):
            try:
                # Prepare the request
                files = {"file": image_file.getvalue()}
                response = requests.post(FASTAPI_URL, files=files, timeout=60)

                # Handle the response
                if response.status_code == 200:
                    result = response.json()

                    # Display retrieved captions
                    retrieved_captions = result.get("retrieved_captions", [])
                    if retrieved_captions:
                        st.subheader("Retrieved Similar Captions:")
                        for idx, caption in enumerate(
                            retrieved_captions, start=1
                        ):
                            st.write(f"{idx}. {caption}")
                    else:
                        st.warning("No similar captions were retrieved.")

                    # Display RAG description
                    rag_description = result.get("rag_description", """No RAG
                                                  description available.""")
                    st.subheader("Generated Caption (RAG-enhanced):")
                    st.write(rag_description)

                    # Display classic description
                    classic_description = result.get(
                                                    "classic_description",
                                                    """No classic
                                                        description
                                                        available.""")
                    st.subheader("Generated Caption (Without RAG):")
                    st.write(classic_description)

                else:
                    st.error(
                        f"""Error {response.status_code}:
                        Unable to process "
                        the image."""
                    )      
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
