import os
from typing import List

import uvicorn
import json_repair
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- 1. Initial Setup ---

# Load environment variables
load_dotenv()
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY")
if not ROUTER_API_KEY:
    raise ValueError("ROUTER_API_KEY not found. Please add it to your .env file.")

# --- 2. OpenAI Client Initialization ---

# Define the model to be used
llm_model = 'openai/gpt-4o-mini'

# Initialize OpenAI client for Requesty.ai
client = openai.OpenAI(
    api_key=ROUTER_API_KEY,
    base_url="https://router.requesty.ai/v1",
)

# --- 3. Data Models (Pydantic) ---

class QuizRequest(BaseModel):
    srcLang: str = Field(..., description="The language in which questions should be generated.")
    words: List[str] = Field(..., min_items=5, max_items=5, description="List of 5 words.")

# --- 4. Prompt Engineering ---

def build_prompt(words: List[str], language: str) -> str:
    word_list = ', '.join(f'"{w}"' for w in words)
    return f"""
You will receive 5 words. Your task is to generate 10 multiple-choice quiz questions (2 per word).

All output must be written in **{language}**.
The Correct Answer must be the char of the correct choice like A, B, C or D
like the following : "correct_answer": "B"

Use a variety of question types, such as:
- Definition of the word.
- Synonym or alternative in the same language.
- Translation of the word.
- Contextual usage.
- Part of speech.

Each question must follow this JSON format:
[
  {{
    "question": "1. Your question here",
    "options": [
      "A. Option one",
      "B. Option two",
      "C. Option three",
      "D. Option four"
    ],
    "correct_answer": "B"
  }},
  ...
]

Rules:
- Return exactly 10 questions.
- All questions must relate to the following words: {word_list}
- All options must be plausible and only one correct.
- Only return a clean valid JSON array. No explanation or extra text.
- Never return a Number in the correct_answer.

Words: {word_list}

```json
"""

# --- 5. Helper Functions ---

def parse_json(text: str):
    """Attempts to repair and parse a JSON string."""
    try:
        return json_repair.loads(text)
    except Exception:
        return None

# --- 6. FastAPI App Setup ---

app = FastAPI()

# Define allowed origins for CORS
origins = [
    "[https://www.turjuman.online](https://www.turjuman.online)",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 7. API Endpoints ---

@app.get("/")
async def read_root():
    """Root endpoint returning basic API info."""
    return {"message": "Welcome ;) Quizzes Generator API For Turjuman is running."}

@app.post("/generate-questions/")
async def generate_questions(request: QuizRequest):
    """API endpoint to generate quiz questions."""
    if len(request.words) != 5:
        raise HTTPException(status_code=400, detail="Exactly 5 words are required.")

    prompt = build_prompt(request.words, request.srcLang)
    print('Prompt has been built.')

    try:
        print("Sending request to the model...")
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        print('Response received from the model.')

        if not response.choices:
            raise Exception("No response choices found from the model.")

        llm_response_content = response.choices[0].message.content
        questions = parse_json(llm_response_content)

        if not questions:
            raise HTTPException(status_code=500, detail="Failed to parse model response as JSON.")

        return questions

    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        raise HTTPException(status_code=502, detail=f"An error occurred with the AI service: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

# --- 8. Run the Application ---

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
