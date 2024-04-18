# %%
pip install -r requirements.txt

# %%
import os
import tiktoken
import numpy as np
import openai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt

# %%

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = "https://testsopenaiajay.openai.azure.com/" #os.getenv('OPENAI_API_BASE')
openai.api_key = "ff5e3e2c133340da83c9e8dcb6d25bb9" #os.getenv("OPENAI_API_KEY")

# Define embedding model and encoding
EMBEDDING_MODEL = 'text-embedding-ada-002'
COMPLETION_MODEL = 'text-davinci-003'
encoding = tiktoken.get_encoding('cl100k_base')


# %%
df = pd.read_csv('./data/movies.csv')
print(df.shape)
df.head()

# %%
# add a new column to the dataframe where you put the token count of the review
df = df.assign(token_count=df['overview'].apply(lambda x: len(encoding.encode(x))))

# print the first 5 rows of the dataframe, then also the total number of tokens
total_tokens = df['token_count'].sum()

cost_for_embeddings = total_tokens / 1000 * 0.0004
print(f"Test would cost ${cost_for_embeddings} for embeddings")

# %%
#@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def get_embedding(text) -> list[float]:
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=text, engine="embeddingada")["data"][0]["embedding"]

# %%
df = df.assign(embedding=df['overview'].apply(lambda x: get_embedding(x)))
df.head()

# %%
# Let's pick a movie that exists in df, keeping in mind we only have 500 movies in it!
movie = "Joker"

# get embedding for movie
e = df[df['original_title'] == movie]['embedding'].values[0]

# get cosine similarity between movie and all other movies and sort ascending
similarities = df['embedding'].apply(lambda x: cosine_similarity(x, e))

# get most similar movies
movies = df.assign(similarity=similarities).sort_values(by='similarity', ascending=False)[['original_title', 'similarity', 'overview']]
movies[0:6]

# %%
df.to_csv('file1.csv')

df1 = pd.read_csv ('file1.csv')

# %%

np.save('embeddings.npy', df)

# %%
embeddings = np.load('embeddings.npy', allow_pickle=True)

df1 = pd.DataFrame(embeddings,columns=['id','original_language','original_title','popularity','release_date','vote_average','vote_count','genre','overview','revenue','runtime','tagline','token_count','embedding'])
# Let's pick a movie that exists in df, keeping in mind we only have 500 movies in it!
movie = "The Italian Job"

# get embedding for movie
e = df1[df1['original_title'] == movie]['embedding'].values[0]

# get cosine similarity between movie and all other movies and sort ascending
similarities = df1['embedding'].apply(lambda x: cosine_similarity(x, e))

# get most similar movies
movies = df1.assign(similarity=similarities).sort_values(by='similarity', ascending=False)[['original_title', 'similarity', 'overview']]
movies[0:6]


