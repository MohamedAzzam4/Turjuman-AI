# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import os
from dotenv import load_dotenv
import json_repair
import json 
import uvicorn
import openai


load_dotenv()
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY")
llm = 'google/gemini-2.5-flash-lite-preview-06-17'

client = openai.OpenAI(
    api_key= ROUTER_API_KEY,
    base_url= "https://router.requesty.ai/v1",
    default_headers= {"Authorization": f"Bearer {ROUTER_API_KEY}"}
)

# --- Pydantic Models ---
# نموذج لمدخلات الـ API
class TranslationInput(BaseModel):
    srcLang: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The source language needed to be translated from."
    )
    
    targetLang: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The Targeted language needed to be translated to."
    )
    
    word: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The word to translate."
    )
    
    context : Optional[str] = Field(
        None,
        min_length=0,
        max_length=1000,
        description="The context paragraph where the word appears., it may contains no context just a simple word need to be transalted"
    )



class Translation(BaseModel):
    translation: str = Field(
        ...,
        min_length=1, 
        max_length=255,
        description="The translated word in the target language , based on the provided context."
    )
    target_synonyms: List[str] = Field(
        ...,
        min_items=0,
        max_items=5,
        description="Different synonymous words in targeted language, relevant to the context. No duplicates."
    )
    source_synonyms: List[str] = Field(
        ...,
        min_items=0, 
        max_items=5,
        description="Synonyms of the word in source_language. No duplicates."
    )
    definition: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Definition of the original word in Source language."
    )
    example_usage: List[str] = Field(
        ...,
        min_items=1,
        max_items=4,
        description="A several examples sentences or phrases using the word to demonstrate its usage in context in source language."
    )


# --- Helper Functions ---

def parse_json(text):
    """Attempts to parse text as JSON, using json_repair as a fallback."""
    try:
        # First, try standard JSON parsing (more strict)
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # If standard parsing fails, try json_repair
        try:
            return json_repair.loads(text)
        except Exception as e:
            print(f"json_repair failed: {e}") 
            return None

def build_translation_messages(word: str, srcLang: str, targetLang: str, context: Optional[str] = None) -> List[dict]:
    """Builds the message list for the Gemini API call."""
   
    schema_json_string = json.dumps(Translation.model_json_schema(), ensure_ascii=False, indent=2)
    user_message = f"srcLang: {targetLang.strip()}, target_Language: {targetLang.strip()} , Word: {word.strip()}"
    if context:
        user_message = f"Context: {context.strip()}\n" + user_message

    messages = [
        {
            "role": "system",
            "content": "\n".join([
                f"You are a professional translator from {srcLang} to {targetLang}.",
                f"You will be provided with a word in {srcLang} and its context is optional.",
                "Translate the word based on the context.",
                "Provide:",
                f"- The translated word in {targetLang} as the user will specify.",
                f"- A list of up to 5 relevant synonyms in {targetLang}.",
                f"- A list of up to 5 relevant synonyms in {srcLang}.",
                f"- The definition of the original word in {srcLang}.",
                f"- A several examples or sentences using the original word in {srcLang}.",
                "Your output must be a valid JSON object exactly matching the following Pydantic schema:",
                f"```json\n{schema_json_string}\n```", # تضمين الـ schema في الـ prompt
                "Do not add any extra text before or after the JSON.",
            ])
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    return messages



# --- FastAPI App Setup ---

app = FastAPI()

origins = [
    "https://www.turjuman.online",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- FastAPI Endpoints ---

@app.get("/")
async def read_root():
    """Root endpoint returning basic API info."""
    return {"message": "Welcome ;) Translator API For Turjuman is running.", "version": "1.0.0"}

@app.post("/translate", response_model=Translation)
async def translate_word_endpoint(input_data: TranslationInput):
    """
    Translates an English word to Arabic based on context using the Turjuman Model.

    Expects a JSON body with 'word' and 'context'.
    Returns a JSON object with translation details, synonyms, definition, and example usage.
    """
    messages = build_translation_messages(input_data.word, input_data.srcLang, input_data.targetLang, input_data.context)
    #print(messages)
    try:

        response = client.chat.completions.create(
                model = llm,
                messages=  messages
            )


        llm_response = response.choices[0].message.content
        raw_json_string = llm_response

 
        parsed_data = parse_json(raw_json_string)

 
        if parsed_data is None:
            print(f"Failed to parse or repair JSON for input: word='{input_data.word}', context='{input_data.context[:50]}...'")
            print(f"Raw response content: {raw_json_string}")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse or repair JSON response from the language model."
            )


        validated_data = Translation(**parsed_data)

        # إرجاع البيانات الصحيحة
        return validated_data

    except ValidationError as e:
        # Catch Pydantic validation errors if the parsed JSON doesn't match the model
        print(f"Pydantic validation error: {e.errors()}")
        print(f"Problematic JSON data: {parsed_data}") 
        raise HTTPException(
            status_code=500,
            detail=f"Turjuman Model returned data that doesn't match the expected structure"
        )
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred:")
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred during translation"
        )
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(app, host="0.0.0.0", port=port)
