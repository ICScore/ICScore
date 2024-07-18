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

# Shuffle and split the data into validation and weight tuning sets
split_data = story_final.copy()
random.shuffle(split_data)
validation_set = split_data[:20]
weight_tuning_set = split_data[20:]

# Ensure directories exist
os.makedirs('./datas/transformed_data', exist_ok=True)
os.makedirs('./datas/validation_set', exist_ok=True)
os.makedirs('./datas/weight_tuning_set', exist_ok=True)

# Process each transformation and story
for transformation in transformations:
    transformation_name = transformation["Transformation"]
    original_example = transformation["Original_Example"]
    transformed_example = transformation["Transformed_Example"]

    transformed_stories = []

    for entry in story_final:
        story = entry["story"]

        # Transform the story
        transformed_story = transform_story(story, transformation_name, original_example, transformed_example)
        
        # Collect the transformed story
        new_entry = {
            "story": story,
            "transformed_story": transformed_story,
        }
        
        transformed_stories.append(new_entry)

    # Write all transformed stories to the main file in the original order
    with open(f'./datas/transformed_data/{transformation_name}_transformed_stories.jsonl', 'w', encoding='utf-8') as file:
        for story in transformed_stories:
            json.dump(story, file, ensure_ascii=False)
            file.write('\n')

    # Create validation and weight tuning sets from the transformed stories based on split_data order
    transformed_validation_set = [next(story for story in transformed_stories if story["story"] == v["story"]) for v in validation_set]
    transformed_weight_tuning_set = [next(story for story in transformed_stories if story["story"] == w["story"]) for w in weight_tuning_set]

    # Write validation set
    with open(f'./datas/validation_set/{transformation_name}_transformed_stories.jsonl', 'w', encoding='utf-8') as file:
        for story in transformed_validation_set:
            json.dump(story, file, ensure_ascii=False)
            file.write('\n')

    # Write weight tuning set
    with open(f'./datas/weight_tuning_set/{transformation_name}_transformed_stories.jsonl', 'w', encoding='utf-8') as file:
        for story in transformed_weight_tuning_set:
            json.dump(story, file, ensure_ascii=False)
            file.write('\n')