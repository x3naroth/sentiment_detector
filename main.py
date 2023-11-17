import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time
import re
import nltk
from textblob import TextBlob

#Scraping
input_csv = 'input.csv'  
output_dir = 'Output_txt'  

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_csv(input_csv)

for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        time.sleep(5)

        title = soup.select_one('title').get_text()

        section = soup.find('div', class_='td-post-content tagdiv-type')

        if section is None:
            section = soup.find('div', class_='tdb-block-inner td-fix-index')

        if section is not None:
            paragraphs = section.find_all('p')
            text = '\n'.join([p.get_text() for p in paragraphs])

            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            output_file = f'{output_dir}/{url_id}.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f'{title}\n\n')
                f.write(f'\n{text}')

            print(f'Guardado: {output_file}')
        else:
            print(f'No se encontró la sección deseada en la página: {url}')

    else:
        print(f'Error al acceder a la URL: {url}')

print('Proceso de scraping completado.')

#NLP analysis
nltk.download('punkt')

def syllable_count(word):
    vowels = "AEIOUaeiou"
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le"):
        count += 1
    if count == 0:
        count += 1
    return count

with open('positive-words.txt', 'r') as file1, open('negative-words.txt', 'r') as file2:
    positive_words = file1.read().splitlines()
    negative_words = file2.read().splitlines()

stop_words = []
for i in range(1, 8):
    with open(f'StopWords_{i}.txt', 'r') as file:
        stop_words += file.read().splitlines()

cleaned_positive_words = [word for word in positive_words if word not in stop_words]
cleaned_negative_words = [word for word in negative_words if word not in stop_words]

scraped_files = os.listdir(output_dir)

results = []

for file in scraped_files:
    file_path = os.path.join(output_dir, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    url_id = file.split('.')[0]

    text_tokens = nltk.word_tokenize(text)
    total_words = len(text_tokens)
    total_sentences = len(nltk.sent_tokenize(text))

    positive_score = sum(1 for word in text_tokens if word in cleaned_positive_words)

    negative_score = (-1 * sum(-1 for word in text_tokens if word in cleaned_negative_words))

    polarity_score = ((positive_score - negative_score) / ((positive_score + negative_score) + 0.000001))

    subjectivity_score = ((positive_score + negative_score) / (total_words + 0.000001))

    average_sentence_length = total_words / total_sentences

    complex_word_count = sum(1 for word in text_tokens if syllable_count(word) > 2)
    percentage_of_complex_words = (complex_word_count / total_words) * 100

    fog_index = 0.4 * (average_sentence_length + percentage_of_complex_words)

    average_words_per_sentence = total_words / total_sentences
    
    personal_pronouns = ['I', 'we', 'my', 'ours', 'us']

    personal_pronoun_count = sum(1 for word in text_tokens if word.lower() in personal_pronouns)

    average_word_length = sum(len(word) for word in text_tokens) / total_words

    syllable_per_word = sum(syllable_count(word) for word in text_tokens)

    result = {
        'URL_ID': file.split('.')[0],
        'URL': df[df['URL_ID'] == url_id]['URL'].values[0],
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score,
        'AVG SENTENCE LENGTH': average_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_of_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': average_words_per_sentence,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': total_words,
        'SYLLABLE PER WORD': syllable_per_word,
        'PERSONAL PRONOUNS': personal_pronoun_count,
        'AVG WORD LENGTH': average_word_length
    }
    results.append(result)

df = pd.DataFrame(results)

df.to_csv('Output_data.csv', index=False)

print('Data analysis completed.')