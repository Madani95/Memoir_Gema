pip install plotly

# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm import OpenAI
from pandasai import SmartDataframe
from pandasai import SmartDatalake
import os
from dotenv import load_dotenv
import matplotlib
import plotly as px
import tkinter as tk
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, dark_theme_css

#Définition du Backend pour les figures de matplotlib
#matplotlib.use("Agg")

# Chargement des variables d'environnement
load_dotenv()
openai_api_key = os.getenv("openai_api_key", 'YOUR_OPENAI_API_KEY')


# Classe personnalisée pour intégrer PandasAI
class CustomPandasAI(PandasAI):
    def chat(self, *args, **kwargs):
        raise NotImplementedError("The 'chat' method has not been implemented yet.")

# Classe pour gérer la conversation avec un fichier CSV
class CSVChat:
    def __init__(self):
        self.response = None
        self.current_plot = None
        self.df = None
        self.prompt_history = []

    # Détecte le séparateur utilisé dans le fichier CSV
    def detect_separator(self, file):
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                file.seek(0)
                first_line = file.readline().decode(encoding)
                comma_count = first_line.count(',')
                semicolon_count = first_line.count(';')
                file.seek(0)
                if comma_count > semicolon_count:
                    return ','
                else:
                    return ';'
            except UnicodeDecodeError:
                pass
        raise ValueError("Could not detect the separator using the provided encodings.")

    # Gérer la conversation avec le DataFrame
    def chat_with_df(self, df, prompt):
        
        st.session_state.openai_api_key = openai_api_key

        llm = OpenAI(api_token=st.session_state.openai_api_key)
        pandas_ai = CustomPandasAI(llm)
        french_prompt = "En français, " + prompt

        result = pandas_ai.run(df, prompt=french_prompt)

        # Gestion des figures
        fig = plt.gcf()
        if fig.get_axes():
            self.current_plot = fig
        plt.close(fig)   

        return result

    # Gérer l'interaction utilisateur pour le chat avec CSV
    def handle_csv_chat(self):
        input_csv = st.file_uploader("Upload your CSV file", type=['csv'], help="Téléchargez un fichier CSV pour analyse.")
        
        # Quand le CSV est chargé
        if input_csv:
            separator = self.detect_separator(input_csv)
            self.df = pd.read_csv(input_csv, sep=separator)
            
            # Afficher le dataframe
            st.dataframe(self.df, use_container_width=True)
            st.write(f"Total Rows: {self.df.shape[0]}, Total Columns: {self.df.shape[1]}")

            # Afficher le type de colonnes du dataframe
            st.write("Column Types: ")
            col_types = pd.DataFrame(self.df.dtypes, columns=['Type']).T
            st.table(col_types)

            # Boite à sélection de colonnes pour afficher les statistiques
            col_to_show = st.selectbox("Select column to show statistics", self.df.columns)
            desc_stats = pd.DataFrame(self.df[col_to_show].describe()).T
            st.table(desc_stats)

            # Séparation de l'affichage de la page en deux colonnes: l'une pour les input et l'autre pour les output
            col1, col2 = st.columns([1, 2])

            with col1:
                with st.form(key='normal_chat_form'):
                    input_text = st.text_area("Enter your query:", help="Entrez une question ou une commande liée au fichier CSV téléchargé.")
                    if st.form_submit_button("Submit"):
                        self.response = self.chat_with_df(self.df, input_text)
                        self.prompt_history.append(input_text)

            with col2:
                if self.response:
                    if self.current_plot:  # Check if there's a plot
                        fig = px.line(self.df)  # Exemple : Utilisez la fonction Plotly Express appropriée ici
                        st.plotly_chart(fig)
                    else:  # Otherwise, display the textual response
                        col2.write(self.response)

            # Affichage de l'historique des promptes
            st.subheader("Prompt history:")
            st.write(self.prompt_history)


# Classe pour traiter les fichiers PDF
class PDFProcessor:
    def __init__(self, pdf_docs):
        self.pdf_docs = pdf_docs
        self._pdf_text = None
        self._text_chunks = None
        self._vectorstore = None

    # Méthode pour extraire le texte du PDF
    @property
    def pdf_text(self):
        if self._pdf_text is None:
            self._pdf_text = self._get_pdf_text()
        return self._pdf_text

    # Méthode pour diviser le texte en morceaux
    @property
    def text_chunks(self):
        if self._text_chunks is None:
            self._text_chunks = self._get_text_chunks()
        return self._text_chunks

    # Méthode pour obtenir le vecteur des textes
    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = self._get_vectorstore()
        return self._vectorstore

    def _get_pdf_text(self):
        text = ""
        for pdf in self.pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def _get_text_chunks(self):
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000,
                                              chunk_overlap=200, length_function=len)
        return text_splitter.split_text(self.pdf_text)

    def _get_vectorstore(self):
        embeddings = OpenAIEmbeddings()
        return FAISS.from_texts(texts=self.text_chunks, embedding=embeddings)


class ChatPDF:
    def __init__(self):
         self.conversation = None
         self.chat_history = []
         self.combined_history = []  # Combine prompts and their responses
            
    # Obtenir la chaîne de conversation
    def get_conversation_chain(self, vectorstore):
        llm = ChatOpenAI(api_key=openai_api_key)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        return ConversationalRetrievalChain.from_llm(llm=llm,
                                                      retriever=vectorstore.as_retriever(),
                                                      memory=memory)

    # Affiche l'historique de la conversation
    def display_chat_history(self):
        """Displays the chat history in expanders."""
        if 'combined_history' in st.session_state:
            for idx, convo in enumerate(st.session_state.combined_history):
                with st.expander(f"Conversation {idx + 1}", expanded=False):
                    st.write(user_template.replace("{{content}}", convo['prompt']), unsafe_allow_html=True)
                    st.write(bot_template.replace("{{content}}", convo['response']), unsafe_allow_html=True)

    # Gérer l'entrée utilisateur
    def handle_user_input(self, user_question):
        # Only make a request if the user's question is new
        if not self.chat_history or (self.chat_history and self.chat_history[-2].content != user_question):
            response = st.session_state.conversation({'question': user_question})
            self.chat_history.extend([{'type': 'user', 'content': user_question},
                                      {'type': 'bot', 'content': response['chat_history'][-1].content}])
        
        st.write(css, unsafe_allow_html=True)  # Write the CSS once

        # Display the current conversation
        for message in self.chat_history:
            if message['type'] == 'user':  # User messages
                st.write(user_template.replace("{{content}}", message["content"]), unsafe_allow_html=True)
            else:  # Bot messages
                st.write(bot_template.replace("{{content}}", message["content"]), unsafe_allow_html=True)

        # Store the combined prompt and response in the history after displaying them
        if 'combined_history' not in st.session_state:
            st.session_state.combined_history = []
        
        st.session_state.combined_history.append({
            'prompt': user_question,
            'response': self.chat_history[-1]['content']  # Last response in chat_history
        })
        
        # Display the chat history
        self.display_chat_history()

    # Gère l'interaction utilisateur pour le chat avec PDF
    def handle_helios_chat(self):
        with st.sidebar:
            st.subheader("Vos documents")
            pdf_docs = st.file_uploader("Téléchargez vos PDFs ici et cliquez sur 'Process'", accept_multiple_files=True,
                                        help="Téléchargez un ou plusieurs fichiers PDF pour analyser et interroger leur contenu.")
            
            if pdf_docs and st.button("Process"):
                pdf_processor = PDFProcessor(pdf_docs)
                vectorstore = pdf_processor.vectorstore
                st.session_state.conversation = self.get_conversation_chain(vectorstore)
                st.info("Les documents PDF ont été traités avec succès !")

        # Utilisation de st.form pour le champ de saisie et le bouton
        with st.form(key="user_input_form"):
            user_question = st.text_area("Posez une question à propos de vos documents:")
            
            # Le bouton de soumission du formulaire
            submit_button = st.form_submit_button("Submit")

            if submit_button and user_question:  # Si le bouton est cliqué ou "Entrée" est pressé
                self.handle_user_input(user_question)

        # Affiche le bouton "Clear History" seulement s'il y a un historique
        if 'combined_history' in st.session_state and st.session_state.combined_history:
            if st.button("Clear History"):
                st.session_state.combined_history.clear()


# Fonction principale pour exécuter l'application              
def main():
    st.write(dark_theme_css, unsafe_allow_html=True) 
    st.title("HELIOS EXPERT ASSISTANT")

   # Disposition et éléments du sidebar
    with st.sidebar:
        st.image("logo Helios.jpg", width=150)

        st.header("Helios Expert Assistant")
        st.text("Choose a Page:",)
        page = st.radio("", ["ChatCSV", "ChatPDF"],
                        help="Sélectionnez une page pour interagir : ChatCSV pour les fichiers CSV ou ChatPDF pour les fichiers PDF.")

        if page == "Helios":
            st.subheader("Vos documents")
            pdf_docs = st.file_uploader("Téléchargez vos PDFs ici et cliquez sur 'Process'", accept_multiple_files=True)
            if pdf_docs:
                st.button("Process")

    if page == "ChatCSV":
        st.subheader("ChatCSV powered by LLM and PandasAI")
        csv_chat = CSVChat()
        csv_chat.handle_csv_chat()

        if st.button("Clear"):
            csv_chat.response = None
            csv_chat.current_plot = None
            csv_chat.df = None
            csv_chat.prompt_history = []

    elif page == "ChatPDF":
        st.subheader("ChatPDF powered by LLM and ChatOpenAI")
        helios_chat = ChatPDF()
        helios_chat.handle_helios_chat()


if __name__ == "__main__":
    main()
