import json
import openai
from dotenv import load_dotenv
import os
import random

# Load .env file
load_dotenv()
# Configure the OpenAI API
OPENAI_API_KEY = os.getenv('API_KEY')
openai.api_key = OPENAI_API_KEY


def transform_story(story, transformation, original_example, transformed_example):
    query = f"""
You are a creative story writer. Given an original story, add {transformation} to make the original story creative and interesting.

For example, 
Original story: {original_example} 
transformed story: {transformed_example}

However, the original story you need to change is this.
Original story: {story}

Add {transformation} to this original story to create an interesting transformed story.
Only provide the transformed story without any titles or extra words. For instance, don't write 'Original story:' or 'Transformed story:', just the story you created.
"""
    print(f"Generated query for first transformation:\n{query}")
    messages = [
        {"role": "system", "content": "You are a creative story writer."},
        {"role": "user", "content": query}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    new_story = response['choices'][0]['message']['content'].strip() if response and response['choices'] else f"Could not generate transformed story for: {story}"

    return new_story

# Load transformations.jsonl
with open('./datas/transformations.jsonl', 'r', encoding='utf-8') as file:
    transformations = [json.loads(line) for line in file if line.strip()]

# Load rocStories.jsonl
with open('./datas/rocStories.jsonl', 'r', encoding='utf-8') as file:
    story_final = [json.loads(line) for line in file if line.strip()]

# Process each transformation and story
for transformation in transformations:
    transformation_name = transformation["Transformation"]
    original_example = transformation["Original_Example"]
    transformed_example = transformation["Transformed_Example"]

    transformed_stories = []

    for i, entry in enumerate(story_final):
        story = entry["story"]

        # First transformation
        new_stories = []
        new_interesting = transform_story(story, transformation_name, original_example, transformed_example)
        new_stories.append(new_interesting)
        
        # Collect the transformed story
        new_entry = {
            "story": story,
            "transformed_story": new_interesting,
        }
        
        transformed_stories.append(new_entry)

    # Write all transformed stories to the main file
    with open(f'./datas/transformed_data/{transformation_name}_transformed_stories.jsonl', 'w', encoding='utf-8') as file:
        for story in transformed_stories:
            json.dump(story, file, ensure_ascii=False)
            file.write('\n')

    # Shuffle and split the data into validation and weight tuning sets
    random.shuffle(transformed_stories)
    validation_set = transformed_stories[:20]
    weight_tuning_set = transformed_stories[20:]

    # Write validation set
    with open(f'./datas/validation_set/{transformation_name}_transformed_stories.jsonl', 'w', encoding='utf-8') as file:
        for story in validation_set:
            json.dump(story, file, ensure_ascii=False)
            file.write('\n')

    # Write weight tuning set
    with open(f'./datas/weight_tuning_set/{transformation_name}_transformed_stories.jsonl', 'w', encoding='utf-8') as file:
        for story in weight_tuning_set:
            json.dump(story, file, ensure_ascii=False)
            file.write('\n')
