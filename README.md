# Image Captioning App with BLIP

This project uses the Hugging Face BLIP model for generating image captions, deployed via a FastAPI backend and a Streamlit frontend.

## Getting Started

### Prerequisites

- Python 3.7+
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mehyarmlaweh/blip-captioning-interface.git
   cd REPO_NAME

2. **Create and activate a virtual environment:**

   ```bash
    python -m venv myenv
    source myenv/bin/activate  # macOS/Linux
    myenv\Scripts\activate     # Windows


3. **Install required packages:**
       ```bash
    pip install -r requirements.txt


4. **Set up your API token in a .env file:**
    HUGGINGFACE_API_TOKEN=hf_your_token_here


### Running the Application


   ```bash
    uvicorn backend:app --host 0.0.0.0 --port 8000
    streamlit run app.py
