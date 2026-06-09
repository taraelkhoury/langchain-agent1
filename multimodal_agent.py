from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
import os


load_dotenv(override=True)


# -------------------------------------------------------
# 1. Create Gemini multimodal model
# -------------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0
)

agent = create_agent(
    model=model,
    system_prompt="You are a science fiction writer. Create and describe futuristic capital cities."
)


# -------------------------------------------------------
# 2. Text input example
# -------------------------------------------------------

text_question = HumanMessage(content=[
    {
        "type": "text",
        "text": "What is the capital of the Moon?"
    }
])

response = agent.invoke(
    {"messages": [text_question]}
)

print("\n--- Text response ---")
print(response["messages"][-1].content)


# -------------------------------------------------------
# 3. Image input example
# -------------------------------------------------------
# Put an image in your project folder and name it moon_city.png
# If the image does not exist, this part will be skipped.

image_path = "moon_city.png"

if os.path.exists(image_path):
    with open(image_path, "rb") as image_file:
        img_b64 = base64.b64encode(image_file.read()).decode("utf-8")

    image_question = HumanMessage(content=[
        {
            "type": "text",
            "text": "Tell me about this futuristic capital city image."
        },
        {
            "type": "image",
            "base64": img_b64,
            "mime_type": "image/png"
        }
    ])

    response = agent.invoke(
        {"messages": [image_question]}
    )

    print("\n--- Image response ---")
    print(response["messages"][-1].content)

else:
    print("\n--- Image response skipped ---")
    print("No image file found. Add an image named moon_city.png to test image input.")


# -------------------------------------------------------
# 4. Audio input example
# -------------------------------------------------------
# Put an audio file in your project folder and name it audio.wav
# If the audio file does not exist, this part will be skipped.

audio_path = "audio.wav"

if os.path.exists(audio_path):
    with open(audio_path, "rb") as audio_file:
        aud_b64 = base64.b64encode(audio_file.read()).decode("utf-8")

    audio_question = HumanMessage(content=[
        {
            "type": "text",
            "text": "Tell me about this audio file."
        },
        {
            "type": "file",
            "source_type": "base64",
            "data": aud_b64,
            "mime_type": "audio/wav"
        }
    ])

    response = agent.invoke(
        {"messages": [audio_question]}
    )

    print("\n--- Audio response ---")
    print(response["messages"][-1].content)

else:
    print("\n--- Audio response skipped ---")
    print("No audio file found. Add an audio file named audio.wav to test audio input.")