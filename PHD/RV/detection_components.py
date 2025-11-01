import os
import time

import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
client = genai.GenerativeModel("gemini-2.0-flash-lite")


def question_generation(entity, content):
    instructs = (
        "I will give you some information about the entity. You should use all this information to generate a question, and the answer to your question is the entity. Do not include the entity in your question.\
  \nThere is an example.\nentity:World War II\ninformation: World War II, also known as the Second World War, was a global war that lasted from 1939 to 1945.\nquestion: which global war lasted from 1939 to 1945?\n\
  entity: {}\ninformation: {}\nquestion:"
    )
    prompt = instructs.format(entity, content)
    response = request_api(prompt)
    return response


def reverse_modeling(question):
    prompt = "You should answer the following question as short as possible. {}"
    prompt = prompt.format(question)
    response = request_api(prompt)
    return response


def request_api(Prompts):
    flag = True
    while flag:
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=500,
            )
            response = client.generate_content(Prompts, generation_config=generation_config)
            text_response = response.text.strip()
            flag = False
            return text_response
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                print("Rate limit exceeded")
                time.sleep(0.01)
            else:
                print("gemini error:", e)
                time.sleep(0.005)


def question_generation_pipeline(entity, content):
    question = question_generation(entity, content)  # construct query
    answer = reverse_modeling(question)
    record = {"entity": entity, "claim": content, "question": question, "answer": answer}

    return record


# if __name__ == '__main__':
#   record = question_generation_pipeline()
