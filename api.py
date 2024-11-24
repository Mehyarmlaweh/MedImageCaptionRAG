"Api"
import json
import os
import base64
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from retrieve import check_image_size_and_dimensions
from retrieve import get_embeddings, retrieve_similar_captions

# Load environment variables
load_dotenv()
# Set the inference profile ID from environment variables
INFERENCE_PROFILE_ID = os.getenv("INFERENCE_PROFILE_ID")
# BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Initialize the Bedrock client
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name='eu-west-3'
)

# Initialize FastAPI
app = FastAPI()


def encode_image_to_base64(image_file: bytes) -> str:
    """
    Encodes an image in binary format to a base64 string.

    Args:
        image_file (bytes): The binary data of the image.

    Returns:
        str: The base64 encoded string of the image.
    """
    return base64.b64encode(image_file).decode("utf-8")


@app.post("/caption/")
async def predict_image_description(file: UploadFile = File(...)):
    """
    Predicts the description of the uploaded medical image by invoking
    a Claude 3.5 Sonnet from bedrock.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        dict: A dictionary containing the predicted description of the image.
    """
    # Read the image file
    image_data = await file.read()

    # Convert the image to base64
    image_base64 = encode_image_to_base64(image_data)
    if not check_image_size_and_dimensions(image_data):
        return {"error": "Image does not meet size or dimension requirements."}

    embedding = get_embeddings(image_data)
    if embedding is None:
        return {"error": "Failed to generate image embeddings."}

        # Retrieve similar captions from Milvus
    similar_captions = retrieve_similar_captions(embedding)
    try:
        # Generate RAG-enhanced description (with retrieval)
        if similar_captions:
            retrieval_context = "\n".join(similar_captions)
            rag_prompt = f"""
                You are a medical assistant with expertise in medical imaging.
                Knowing that similar images have these captions:
                {retrieval_context}
                Write a detailed description of the input image, highlighting
                anatomical structures, abnormalities, and potential diagnoses.
                """
            rag_payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_base64
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": rag_prompt
                                }
                            ]
                        }
                    ],
                    "max_tokens": 1000,
                    "anthropic_version": "bedrock-2023-05-31"
            }
            rag_response = bedrock_client.invoke_model(
                    modelId=INFERENCE_PROFILE_ID,
                    contentType="application/json",
                    body=json.dumps(rag_payload)
            )
            rag_output_binary = rag_response["body"].read()
            rag_output_json = json.loads(rag_output_binary)
            rag_description = rag_output_json["content"][0]["text"]
        else:
            rag_description = '''No similar captions
                                retrieved for RAG enhancement.'''

        # Prepare the payload for Bedrock model
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": '''You are a medical assistant
                            with expertise in medical imaging. Given
                            the uploaded medical image, provide a detailed
                            description that includes key anatomical
                            structures,
                                any abnormalities or conditions visible,
                                and potential diagnoses or observations
                                relevant to the image. Be precise and clear
                                    in your explanation, highlighting
                                    significant
                                    features of the image'''
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "anthropic_version": "bedrock-2023-05-31"
        }

        # Call the Bedrock model to get the prediction
        response = bedrock_client.invoke_model(
            modelId=INFERENCE_PROFILE_ID,
            contentType="application/json",
            body=json.dumps(payload)
        )

        # Extract the result
        output_binary = response["body"].read()
        output_json = json.loads(output_binary)
        classic_description = output_json["content"][0]["text"]

        return {
                "classic_description": classic_description,
                "rag_description": rag_description,
                "retrieved_captions": similar_captions
            }

    except Exception as e:
        return {"error": str(e)}
# uvicorn api:app --reload
