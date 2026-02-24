# front-end
import streamlit as st

# back-end
import random
import time
import sys

# doc treatment
import PyPDF2
import filetype

#stats and visuals
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from wordcloud import WordCloud

# NLP
import nltk
nltk.data.path.append("./nltk_data")
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.util import ngrams

import spacy
#from collections import Counter
#spacy.cli.download('en_core_web_sm')

# LLM, imports only on local setting 
# import ollama

# for export formats
import json

# Check if running locally (ollama available)
is_local = sys.executable.startswith('/usr/local') or 'localhost' in sys.executable or 'C:\\' in sys.executable

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
		kind = filetype.guess(source.getvalue())
		if kind and kind.mime == 'application/pdf':
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

if not is_local:
    st.write("Exporting to bilingual XLIFF requires xml library; local only for now")

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

st.write("#### Classic NLP tasks")
left_column, right_column, third_column = st.columns(3)

button_parts = left_column.button('Show random parts of books')
button_phrases = right_column.button('Show most common words and phrases')
button_names = third_column.button('Show names and entities')

# buttons behaviour
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
	text = ' '.join(value_words)
	wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis('off')

	st.pyplot(plt)

	#experimenting with bigrams
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

if button_names:
    with st.spinner("Running named entity recognition..."):
        # Creating spaCy language object
        import spacy
        from collections import Counter
        

        nlp = spacy.load("en_core_web_sm")
        nlp.max_length = len(content) + 1

        @st.cache_data
        def spacy_object(content):
            return nlp(content)

        doc = spacy_object(content)

        entities = [(ent.text, ent.label_) for ent in doc.ents]
        persons = [ent for ent in entities if ent[1] == "PERSON"]
        places = [ent for ent in entities if ent[1] == "GPE"]
        organizations = [ent for ent in entities if ent[1] == "ORG"]

        most_common_persons = Counter(persons).most_common(20)
        most_common_places = Counter(places).most_common(20)
        most_common_orgs = Counter(organizations).most_common(20)

        chosen_ent = st.multiselect(
            "Do you want to list ",
            ["persons", "places", "organizations"]
        )

        st.write(entity for entity in most_common_persons)
        st.write(entity for entity in most_common_places)
        st.write(entity for entity in most_common_orgs)

    if chosen_ent:
        st.markdown(f"### Main {chosen_ent} in text:")
        for entity in chosen_ent:
            if entity == "persons":
                for person, count in most_common_persons:
                    st.markdown(f"**{person}** (Count: {count})")
            elif entity == "places":
                for place, count in most_common_places:
                    st.markdown(f"**{place}** (Count: {count})")
            elif entity == "organizations":
                for org, count in most_common_orgs:
                    st.markdown(f"**{org}** (Count: {count})")

# adding experimental LLM layer
st.divider()
st.write("#### LLM dialog (experimental)")

# Check if running locally (ollama available)
#is_local = sys.executable.startswith('/usr/local') or 'localhost' in sys.executable or 'C:\\' in sys.executable

if not is_local:
    st.warning("⚠️ This feature is only available when running locally (requires Ollama)")
    st.text_area("Ask a question about the book", placeholder="e.g. Who is the main character?", height=100, disabled=True)
    st.button("Submit", disabled=True)
else:
    import ollama
    user_prompt = st.text_area("Ask a question about the book", placeholder="e.g. Who is the main character?", height=100)
    button_llm_dialog = st.button("Submit")
    if button_llm_dialog and user_prompt:
        prompt = f"""You are a book editor. You are great at getting reliable insights from a book and interpreting them. 
Use ONLY the following book text to answer the question.

Book text:
{content}

Question: {user_prompt}

Answer based only on the book text above:"""

        with st.spinner("Thinking..."):
            response = ollama.chat(
                model="gemma2:2b",
                messages=[{"role": "user", "content": prompt}],
                options={"num_ctx": 128000}
            )
        st.write(response["message"]["content"])

# adding hybrid NLP/LLM workflows
st.divider()
st.write("""
#### Hybrid ideas (NLP+LLM, in progress)

##### A. Character extraction (classic NLP) + top 5 characters' medaillons (LLM)

spaCy entity recognition > filter persons > filter top 5 > extract chunks of texts surrounding given character > feed chunk corpurs to LLM (RAG) > prompt LLM to create character's medailon

#### B. Talk with actual characters
feed LLM 1) detailed medailons + 2) possibly extracted chunk corpus + 3) extracted character's direct speech (separate workflow) > system prompt as model's personality > talk to model/character
""")

# A. Character Medallion Generation
if not is_local:
    st.warning("⚠️ Character medallion generation requires Ollama (local only)")
    st.selectbox("Select character", ["Gall"], disabled=True)
    st.button("Generate Medallion", disabled=True, key="btn_medallion_generate_disabled")
    st.button("Export character as JSON", disabled=True, key="btn_export_json_disabled")
else:
    import ollama
    import re
    import json
    from collections import Counter
    
    st.subheader("A. Generate Character Medallion")
    
    # Extract top characters
    if 'doc' not in locals():
        with st.spinner("Analyzing characters..."):
            nlp = spacy.load("en_core_web_sm")
            nlp.max_length = len(content) + 1
            doc = nlp(content)
    
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    persons = [ent for ent in entities if ent[1] in ["PERSON", "ORG"]]
    top_characters = Counter(persons).most_common(10)
    character_names = [name for (name, _), count in top_characters]
    
    selected_char = st.selectbox("Select character", character_names)
    
    # Function to generate medallion (outside button logic)
    def generate_medallion(character_name):
        # Extract character chunks
        char_chunks = [content[max(0, pos-300):min(len(content), pos+300)] 
                      for pos in range(len(content)) 
                      if content.startswith(character_name, pos)]
        
        combined_chunks = "\n\n---\n\n".join(char_chunks[:20])
        
        prompt = f"""You are a literary analyst. Based ONLY on these text excerpts about {character_name}, create a character medallion.
Text excerpts:
{combined_chunks}
Focus on personality, role in story, characteristics, keep plot unspoilered.
"""
        
        response = ollama.chat(
            model="gemma2:2b",
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": 128000}
        )
        
        st.session_state['medallion'] = response["message"]["content"]
        st.session_state['character'] = character_name
        st.session_state['chunks'] = char_chunks
        
        return response["message"]["content"]
     
    # Buttons side by side
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Medallion", key="btn_medallion_generate"):
            with st.spinner(f"Generating medallion for {selected_char}..."):
                medallion = generate_medallion(selected_char)
    
    with col2:
        if st.button("Export character as JSON", key="btn_export_json"):
            # Generate medallion if not already in session
            if 'medallion' not in st.session_state or st.session_state.get('character') != selected_char:
                with st.spinner(f"Generating medallion for {selected_char}..."):
                    generate_medallion(selected_char)
            
            # Get book title safely
            book_title = uploaded_file.name.replace('.txt', '').replace('.pdf', '') if uploaded_file else "Unknown Book"
            
            # Build export structure
            character_data = {
                "metadata": {
                    "book_title": book_title
                },
                "characters": [
                    {
                        "name": st.session_state['character'],
                        "profile": st.session_state['medallion']
                    }
                ]
            }
            
            json_str = json.dumps(character_data, ensure_ascii=False, indent=2)
            
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{selected_char}_profile.json",
                mime="application/json"
            )
    
    # Display medallion if it exists (OUTSIDE columns)
    if 'medallion' in st.session_state:
        st.markdown(f"### Character Medallion: {st.session_state['character']}")
        st.write(st.session_state['medallion'])
            
# B. Chat with Character
st.divider()
if not is_local:
    st.warning("⚠️ Character chat requires Ollama (local only)")
    st.text_input("You:", ["What do you think about Helena, Dr. Gall? Be honest."], disabled=True)
    st.button("Talk to them", disabled=True)
else:
    import ollama
    import re
    
    st.subheader("B. Chat with Character")
    
    if 'medallion' in st.session_state and 'character' in st.session_state:
        character = st.session_state['character']
        medallion = st.session_state['medallion']
        char_chunks = st.session_state['chunks']
        
        # Extract dialogues
        quotes = r'[""\"\'\`\«\»\‹\›]([^""\"\'`\«\»\‹\›]+)[""\"\'\`\«\»\‹\›]'
        play = r'^[A-Z\s]{2,}\.\s+([^.\n]+\.)'
        pattern = f'{quotes}|{play}'
        all_dialogues = re.findall(pattern, content, re.MULTILINE)
        all_dialogues = [m[0] or m[1] for m in all_dialogues if any(m)]
        
        # Filter character dialogues
        char_dialogues = []
        for dialogue in all_dialogues:
            pos = content.find(dialogue)
            if pos > -1:
                context = content[max(0, pos-100):pos+len(dialogue)+100]
                if character in context:
                    char_dialogues.append(dialogue)
        
        # Initialize chat
        if 'messages' not in st.session_state:
            system_prompt = f"""You are {character}. Your personality: {medallion}
Story context: {char_chunks[:10]}
Your speech style: {char_dialogues[:30]}
Stay in character and match your speech patterns."""
            st.session_state['messages'] = [{"role": "system", "content": system_prompt}]
        
        # Display chat history
        for msg in st.session_state['messages'][1:]:  # Skip system prompt
            if msg['role'] == 'user':
                st.chat_message("user").write(msg['content'])
            else:
                st.chat_message("assistant").write(f"**{character}**: {msg['content']}")
        
        # Chat input
        user_input = st.chat_input(f"Talk to {character}...")
        
        if user_input:
            st.session_state['messages'].append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            with st.spinner(f"{character} is thinking..."):
                response = ollama.chat(
                    model="gemma2:2b",
                    messages=st.session_state['messages'],
                    options={"num_ctx": 128000}
                )
            
            answer = response["message"]["content"]
            st.session_state['messages'].append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(f"**{character}**: {answer}")
    else:
        st.info("👆 First generate a character medallion above to start chatting!")