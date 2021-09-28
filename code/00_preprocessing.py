"""
Quick look at the data and how clean/dirty it is
"""
import pandas as pd
import re
import spacy


datapath = 'data/raw/train.csv'
raw_data = pd.read_csv(datapath)

raw_data.shape
raw_data.columns

raw_data.head()

raw_data.isna().sum(axis=0) / raw_data.shape[0] * 100

# We will be using text and target. It makes no sense to use id, location
# has plenty of NA, and keyword... well. Lets keep it for now

columns_to_drop = ['id', 'location']
raw_data.drop(columns=columns_to_drop, axis=0, inplace=True)

# Now if we're planning to use keyword we will need to drop NA's as it
# don't make sense imputing in this kind of problem. Lets see what we 
# acctually have in that column

raw_data['keyword'].unique()

# Ok so in fact, we might have some usefull information here. First of all,
# let's drop the NA's. They represent less than 1%.
# We will also need to clean those words a lil bit. For now, let's just add
# them to the general text

raw_data.dropna(axis=0, inplace=True)

# Just in case, create a new column
raw_data['full_text'] = raw_data['keyword'] + ' ' +  raw_data['text']

# Now into the actual text cleaning. We want to get rid of a lot of stuff
# Since this will be needed for the production stage, let's create all the
# cleaning inside functions.

# We want to:
  # Get rid of html tags
  # delete emojis
  # delete urls
  # Perform the normal NLP tasks: lemmatization, to lower, delete stopwords...

# This could all be done in one def, but for the sake of modualrity, let's
# do it one step at a time

def remove_html(string):
    html_pattern = re.compile('<.*?>') 
    clean_string = html_pattern.sub('', string) 
    
    return clean_string

remove_html('<body>Hello there</body>')  # works

def remove_emojis(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # general emotes
                               u"\U0001F300-\U0001F5FF"  # symbols
                               u"\U0001F680-\U0001F6FF"  # transport
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002500-\U00002BEF"  # chinesse stuff
                               u"\U00002702-\U000027B0"  # rest of emojis
                               u"\U00002702-\U000027B0"  
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  
                               u"\u3030"
                               "]+", 
                               flags=re.UNICODE) 
        
    clean_string = emoji_pattern.sub('', string)
    
    return clean_string

remove_emojis('Hello there👽')  # works

# Now remove urls
def remove_url(string):
    
    clean_string = re.sub(r'http|:.*$', '', string)
    
    return clean_string

remove_url('Hello, visit  http://t.co/0DlGChTBIx')  # works

# Finally, before the NLPshy stuf, remove nonalphanum
def remove_non_alphanumeric(string):
    
    cleaned_string = re.sub(r'[^A-Za-z0-9 ]+', '', string)
    
    return cleaned_string

remove_non_alphanumeric('Ha%l%o!``') # also works

# Ok, for the NLP parta, load the dictionary first
nlp_english = spacy.load('en_core_web_md')

# Now build the function. It will take the nlp analyzer as an argument
def nlp_treat_text(string, analyzer):
    
    analysed_text = analyzer(string)  # creates the nlp object

    # Gets rid of punctuation and stopwords. Also lowers it
    text = [
        token.text.lower() for token in analysed_text 
                           if not token.is_punct 
                           and not token.is_stop
                           ]

    clean_text = ' '.join(text)

    return clean_text

sample = raw_data['text'][0:9]

sample.map(lambda x: nlp_treat_text(x, nlp_english))  # Ok so it works


# To end this part, we will build a general function to perform all this
# cleaning tasks

def clean_full_text(text, analyzer):
    
    text = remove_html(text)
    text = remove_emojis(text)
    text = remove_url(text)
    
    # To remove usernames
    text = re.sub(r'@[A-z]+', '', text)
    
    text = remove_non_alphanumeric(text)
    text = nlp_treat_text(text, analyzer)
    
    return text


# Let's try this over the small sample
sample.map(lambda x: clean_full_text(text=x, analyzer=nlp_english))

# It works

# Now we'll reset the index of the df, then clean the text, and then save
# the data
raw_data.reset_index(drop=True, inplace=True)

raw_data['clean_text'] = raw_data['text'].map(
    lambda x: clean_full_text(text=x, analyzer=nlp_english)
    )

destination_path = 'data/clean/clean_data.csv' 
raw_data[['clean_text', 'target']].to_csv(destination_path, encoding='utf-8')