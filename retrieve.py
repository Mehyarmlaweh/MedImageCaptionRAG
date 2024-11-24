"retrievefrom milvus"
import io
import base64
import json
from PIL import Image
import boto3
from pymilvus import connections, Collection
from botocore.exceptions import BotoCoreError, ClientError


TABLE_NAME = 'image_caption_embeddings'
MAX_FILE_SIZE_MB = 25
MAX_DIMENSIONS = (4096, 4096)
MAX_PIXEL_COUNT = 2048 * 2048 * 3
MIN_DIMENSIONS = (256, 256)

# Constants for image limits (Titan model)


def check_image_size_and_dimensions(image_bytes: bytes) -> bool:
    """
    Check if the image meets the size and dimension constraints.
    Returns True if valid, False if image is too large or
    has invalid dimensions.
    """
    # Open image using PIL
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    # Check file size (max 25 MB)
    image_size_mb = len(image_bytes) / (1024 * 1024)
    if image_size_mb > MAX_FILE_SIZE_MB:
        return False

    # Check pixel count (max 2048 * 2048 * 3)
    pixel_count = width * height * 3  # 3 channels (RGB)
    if pixel_count > MAX_PIXEL_COUNT:
        return False
    # Check image dimensions (min 256x256, max 4096x4096)
    if width < MIN_DIMENSIONS[0] or height < MIN_DIMENSIONS[1]:
        return False
    if width > MAX_DIMENSIONS[0] or height > MAX_DIMENSIONS[1]:
        return False

    return True


def connect_to_milvus(host='localhost', port='19530'):
    """
    Connect to the standalone Milvus server only if not already connected.
    """
    try:
        # Check if a connection with the alias 'default' already exists
        if connections.has_connection(alias="default"):
            print("Already connected to Milvus.")
            return

        # Establish a new connection if not connected
        connections.connect(alias="default", host=host, port=port)
        print(f"Connected to Milvus at {host}:{port}")
    except ConnectionError as e:
        print(f"Failed to connect to Milvus: {e}")


def retrieve_similar_captions(query_embedding, limit=3):
    """
    Retrieve the top N similar captions from Milvus
    for a given query embedding.

    Args:
        query_embedding (list): The query embedding to search
        for similar entries.
        collection_name (str): Name of the Milvus collection.
        limit (int): Number of top similar results to retrieve. Default is 3.

    Returns:
        list: A list of dictionaries containing the IDs
        and captions of similar entries.
    """
    try:
        # Connect to Milvus
        connect_to_milvus()
        # Access the collection
        collection = Collection(TABLE_NAME)

        # Define search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        # Perform the search
        results = collection.search(
            data=[query_embedding],         # Query embedding
            anns_field="embedding",
            param=search_params,
            limit=limit,                    # Retrieve top N similar results
            output_fields=["caption"]
        )

        # Debugging: Log raw results
        if not results:
            print("No results found in Milvus.")
            return []

        similar_captions = []
        for hits in results:
            for hit in hits:
                similar_captions.append(hit.caption)  # Access caption directly

        if not similar_captions:
            print("No captions retrieved.")
        else:
            print(f"Retrieved captions: {similar_captions}")

        return similar_captions

    except Exception as e:
        print(f"Error retrieving similar captions: {e}")
        return []


def get_embeddings(image_bytes):
    """
    Generate embeddings for an image using AWS Bedrock
    after checking image constraints.
    """
    try:
        # Check if the image meets the size and dimension constraints
        if not check_image_size_and_dimensions(image_bytes):
            print(f"""Skipping image {image_bytes}:
                   Does not meet size/dimension constraints""")
            return None  # Skip the image if it doesn't meet the requirements

        # Encode image in Base64
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
        except TypeError as te:
            print(f"Error encoding image to Base64: {te}")
            return None

        # AWS Bedrock client
        bedrock = boto3.client('bedrock-runtime', region_name='eu-west-3')

        # Prepare payload
        payload = {"inputImage": base64_image}

        # Invoke Bedrock model
        try:
            response = bedrock.invoke_model(
                modelId="amazon.titan-embed-image-v1",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload)  # Serialize the payload
            )
        except (BotoCoreError, ClientError) as aws_error:
            print(f"AWS Bedrock invocation failed: {aws_error}")
            return None

        # Parse and return embeddings
        try:
            response_payload = json.loads(response['body'].read())
            return response_payload.get('embedding')
        except json.JSONDecodeError as je:
            print(f"Error decoding JSON response: {je}")
        except KeyError as ke:
            print(f"Missing key in response: {ke}")

        return None

    except Exception as e:
        print(f"Unexpected error processing image {image_bytes}: {e}")
        return None
