import streamlit as st
import pandas as pd

import PyPDF2
import filetype

import random
import time

st.subheader('BOOK GLANCER')
st.write('Quick overview of a book and its translation. Random snap, word frequency, n-grams. By default \'R.U.R\' by Karel Čapek on open license. Strictly non-commercial use.')

uploaded_file = st.sidebar.file_uploader('Choose a book', type=['pdf', 'txt'], key=42)
uploaded_file_cs = st.sidebar.file_uploader('Choose a translation', type=['pdf', 'txt'], key=24)

def progress_bar():
	progress_text = 'Reading thru the books.'
	my_bar = st.progress(0, text=progress_text)

	for percent_complete in range(100):
		time.sleep(0.01)
		my_bar.progress(percent_complete + 1, text=progress_text)
	time.sleep(1)
	my_bar.empty()

#upload new book here
if uploaded_file is not None:
	progress_bar()
	def check_type(source):
		kind = filetype.guess(source)
		if kind.mime == 'application/pdf':
			reader = PyPDF2.PdfReader(source)
			content = ''
			for page in reader.pages:
				content += page.extract_text()
		else:	
			content = source.getvalue().decode('utf-8')
		return content
			
	content = check_type(uploaded_file)
	if uploaded_file_cs is not None:
		content_cs = check_type(uploaded_file_cs)
	else:
		content_cs = None
elif uploaded_file is None:
	text = 'sources/rur.txt'
	text_cs = 'sources/rur_cs.txt'

	def open_book(book):
		with open(book, 'r', encoding='utf-8') as text:
			content = text.read()
			return content
		
	content = open_book(text)
	content_cs = open_book(text_cs)
    
#print random middle part of the book and translation
@st.cache_data
def middle_slice(book):
	random_words = random.randint(round(len(book)/4), round(len(book) - len(book)/4))
	return book[random_words:random_words+600]

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

import matplotlib.pyplot as plt, seaborn as sns

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng') 

@st.cache_data
def get_tokens(content):
	tokens = word_tokenize(content)
	return tokens

tokens = get_tokens(content)
if content_cs is not None:
	tokens_cs = get_tokens(content_cs)

lemmatizer = WordNetLemmatizer()

@st.cache_data
def clean_words(tokens):
	#some data cleaning
	#tokenizing text + removing stopwords + lemmatizing
	stops = stopwords.words('english') + ['said', 'saw', 'see', 'copyright', 'u', 'looked', 'made', 'got', 'asked']
	clean_tokens = [token for token in tokens if token.lower() not in stops and token.isalnum()]
	lemmas = [lemmatizer.lemmatize(token) for token in clean_tokens]
	stop_lemmas = ['look', 'know', 'u', 'ask', 'go', 'get', 'make', 'way']
	clean_lemmas = [lemma for lemma in lemmas if lemma not in stop_lemmas]
	tagged = pos_tag(clean_lemmas)

	#some extra cleaning
	#filtering only common nouns, adjectives and verbs

	value_tags = ['NN', 'NNS', 'JJ', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
	value_words = [word[0] for word in tagged if word[1] in value_tags]
	return value_words

value_words = clean_words(tokens)
freq = FreqDist(value_words)

# Front-end behaviour

st.cache_data.clear()
st.sidebar.markdown(f'''
### Original Text Statistics
- **Words**: {len(tokens)}
- **Characters**: {len(content)}
- **Standard Pages**: {len(content) / 1800:.2f}
''')

if content_cs is not None:
	st.sidebar.markdown(f'''
	### Translation Statistics
	- **Words**: {len(tokens_cs)}
	- **Characters**: {len(content_cs)}
	- **Standard Pages**: {len(content_cs) / 1800:.2f}
	''')

left_column, right_column, third_column = st.columns(3)

button_parts = left_column.button('Show random parts of books')
button_phrases = right_column.button('Show most common words and phrases')
button_names = third_column.button('Show names and entities')

if button_parts:
	progress_bar()
	st.cache_data.clear()
	st.write('SOME RANDOM PART OF THE BOOK\n\n', middle_slice(content),
		'\n\n*******************************\n')
	if content_cs is not None:
		st.write('SOME RANDOM PART OF THE TRANSLATION\n\n', middle_slice(content_cs))

if button_phrases:
	progress_bar()

	#plotting into bars
	freq_dct_commons = dict(freq.most_common(25))

	sns.barplot(x=list(freq_dct_commons.values()), y=list(freq_dct_commons.keys()), palette='deep', orient='h')
	plt.title('some most common words')

	st.pyplot(plt)
	
	#as a word cloud
	from wordcloud import WordCloud

	text = ' '.join(value_words)
	wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis('off')

	st.pyplot(plt)

	#experimenting with bigrams

	from nltk.util import ngrams

	#creating bigrams: 2word phrases + their frequency

	grams = list(ngrams(value_words, 2))
	joined_grams = [' '.join(gram) for gram in grams]
	grammy_freq = FreqDist(joined_grams)

	freq_bigrams = dict(FreqDist(grams))
	print(grammy_freq.most_common(20), '\n')

	#plotting underscored phrases into word cloud
	grams_conc = ['_'.join(gram) for gram in grams]
	text = ' '.join(grams_conc)

	wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(text)

	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis('off')
	plt.title('some most common phrases')

	st.pyplot(plt)

import spacy
from collections import Counter
spacy.cli.download("en_core_web_sm")

# Creating spaCy language object
nlp = spacy.load('en_core_web_sm')
nlp.max_length = len(content)+1

@st.cache_data
def spacy_object(content):
	doc = nlp(content)
	return doc

doc = spacy_object(content)

if 'show_entities' not in st.session_state:
    st.session_state.show_entities = False

if button_names:
    st.session_state.show_entities = True

if st.session_state.show_entities:
	entities = [(ent.text, ent.label_) for ent in doc.ents]
	persons = [ent[0] for ent in entities if ent[1] == 'PERSON']
	num_persons = st.selectbox("How many persons", ['most common', 'all'])
       
	map_persons = {'most common': 20, 'all': len(persons)}
	most_common_persons = Counter(persons).most_common(map_persons[num_persons])
	st.markdown('### Persons in Text:')
	for person in most_common_persons:
		st.markdown(f'**{person[0]}: {person[1]}**')
