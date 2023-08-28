import openai
from Models.RAG.RagModel import *
from Models.RAG.RagModel import RAG_target_paragraph_generator
import argparse
import apiKeys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="RAG")
    args = parser.parse_args()

    if args.model == "RAG":
        generator = RAG_target_paragraph_generator("", "")

    openai.api_key = apiKeys.openai_key
    # If memory matters
    # Initialize conversation history
    conversation_history = []
    counter = 0
    while True:
        # Get user input
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # Add user message to conversation history
        conversation_history.append(
            {"role": "user", "content": generator.prompt_generator(user_input)}
        )

        # Make API call
        api_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=conversation_history
        )

        # Extract and print assistant's reply
        assistant_reply = api_response.choices[0].message["content"]
        print("Assistant:", assistant_reply)

        # Add assistant's reply to conversation history
        conversation_history.append({"role": "assistant", "content": assistant_reply})
