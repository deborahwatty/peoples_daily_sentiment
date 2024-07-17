import openai
from openai import OpenAI
import pandas as pd
from collections import defaultdict
import time
from datetime import datetime
import json
import ast
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
from dotenv import load_dotenv
from matplotlib import font_manager

font_prop = font_manager.FontProperties(family='SimHei')


load_dotenv()

genai.configure(api_key=os.getenv("GENAI_API_KEY"))

openai.api_key = os.getenv("OPENAI_API_KEY")

assistant_playground_id = os.getenv("GENAI_FLASH_PLAYGROUND_ID")


class SentimentAssistant:

    def __init__(self, llm, num_entities, df=None):  # Changed from filepath to df
        self.df = df  # Remove the file reading line

        if llm == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.assistant = self.client.beta.assistants.retrieve(assistant_playground_id)
            if num_entities == 1:
                self.playground_id = assistant_playground_id

    def wait_on_run(self, run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run

    '''
    this function is for running an entire dataframe searching for the SAME SINGLE ENTITY in all
    texts contained in the df! 
    entity must be a string 
    
    returns a dictionary, can be used as input for the plot function 
    '''
    def run_single_openai_fulldf(self, entity):

        results = defaultdict(dict)


        for index, row in self.df.iterrows():
            print('Processing ' + str(index))
            text = row['title'] + "\n" + row['doc_content']
            filename = row['filename']
            date = row['timeseq_not_before']
            date = datetime.strptime(str(date), '%Y%m%d').date()

            results[filename]['date'] = date

            thread = self.client.beta.threads.create()

            prompt = f"We are going to analyze the following text: \"{text}\" \nYour task is to assign a sentiment label that the text communicates regarding \"{entity}\", according to the system instructions for the assistant."

            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )

            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id,
            )

            run = self.wait_on_run(run, thread)

            messages = self.client.beta.threads.messages.list(thread_id=thread.id)

            response = json.loads(messages.model_dump_json())

            results[filename]['response'] = response['data'][0]['content'][0]['text']['value']
            results[filename]['messages'] = response
            results = self.extract_labels_and_add_to_dict(results)

        return results

    '''
        this function is for running an entire dataframe searching for the SAME SINGLE ENTITY in all
        texts contained in the df! 
        entity must be a string 

        returns a dictionary, can be used as input for the plot function 
        '''

    def run_single_gemini_fulldf(self, entity):

        # Create the model
        # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
        generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            # safety_settings = Adjust safety settings
            # See https://ai.google.dev/gemini-api/docs/safety-settings
            system_instruction="You are a helpful assistant designed to output sentiment classification labels. All questions are about entity-wise sentiment analysis on Chinese texts. You will analyze the sentiment toward the given volitional entity, inspecting a Chinese text that is provided as the introduction.  The reply should be the assigned label, one of `['Positive-Standard',  'Positive-Slight', 'Neutral', 'Negative-Slight', 'Negative-Standard' ]`. 'Neutral' is the label assigned when you cannot identify any sentiment toward the entity in question. 'Positive-Slight' and 'Negative-Slight' are used if an entity receives slight, vague or uncertain sentiment. Otherwise, the 'Positive-Standard' and  'Negative-Standard' labels are used for all clear sentiments expressed towards the entity. You should not refer to common knowledge about an entity, but strictly analyze the sentiment conveyed in the given text. If both positive or negative sentiments exist, you must decide what is the prevalent or overall strongest sentiment conveyed in the text regarding the entity in question. \nThe output should be a JSON formatted formatted in the following schema: \n{\\n\\t\"label\": string // The label assigned to the entity in question, one of ['Positive-Standard',  'Positive-Slight', 'Neutral', 'Negative-Slight', 'Negative-Standard' ]. If you could not find the entity in the text, write 'none'. \\n}",
        )

        results = defaultdict(dict)

        for index, row in self.df.iterrows():
            print('Processing abc ' + str(index))
            text = row['title'] + "\n" + row['doc_content']
            filename = row['filename']
            date = row['timeseq_not_before']
            date = datetime.strptime(str(date), '%Y%m%d').date()

            results[filename]['date'] = date

            chat_session = model.start_chat(
                history=[
                ]
            )

            prompt = f"We are going to analyze the following text: \"{text}\" \nYour task is to assign a sentiment label that the text communicates regarding \"{entity}\", according to the system instructions for the assistant." #+ "Note that the entity name is given in English, while it will likely appear in Chinese in the text."

            response = chat_session.send_message(prompt)

            results[filename]['response'] = response.text
            results[filename]['messages'] = response
            results = self.extract_labels_and_add_to_dict(results)
        return results



    """
    takes results from run_single_openai_fulldf for a specific entity 
    """
    def plot(self, results, entity_name):

        df_for_plot = self.df.copy()

        # Example of accessing the updated defaultdict
        for filename, data in results.items():
            print(f"{filename}: {data}")

        df_for_plot['sentiment_label'] = df_for_plot['filename'].apply(lambda x: results[x]['label'])
        df_for_plot['date'] = df_for_plot['filename'].apply(lambda x: results[x]['date'])

        # Map sentiment labels to numerical values
        sentiment_mapping = {
            'Positive-Standard': 2,
            'Positive-Slight': 1,
            'Neutral': 0,
            'Negative-Slight': -1,
            'Negative-Standard': -2
        }

        df_for_plot['sentiment_value'] = df_for_plot['sentiment_label'].map(sentiment_mapping)
        #df_for_plot['date'] = pd.to_datetime(df_for_plot['timeseq_not_before'])

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(df_for_plot['date'], df_for_plot['sentiment_value'], marker='o')
        plt.xlabel('Date')
        plt.ylabel('Sentiment')
        plt.title('Sentiment toward ' + entity_name + ' over time', fontproperties=font_prop)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.yticks(range(-2, 3),
                   ['Negative-Standard', 'Negative-Slight', 'Neutral', 'Positive-Slight', 'Positive-Standard'])
        plt.tight_layout()

        # Return the figure object
        return plt.gcf()
    def extract_labels_and_add_to_dict(self, results):
        # Process each entry
        for filename, data in results.items():
            # Parse the string as a dictionary
            response_dict = ast.literal_eval(data['response'])

            print("AAAAAAAAAAAA")
            print(response_dict.keys())
            print("AAAAAAAAAAAA")

            # Extract the label
            label = response_dict['label']
            # Add the label to the original dictionary
            data['label'] = label

        #for filename, data in results.items():
        #    print(data['label'])
        return results
