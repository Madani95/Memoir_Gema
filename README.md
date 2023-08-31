# Memoir_Gema
## HELIOS Expert Assistant

This is a Streamlit application that allows users to interact with their data in a conversational way. The application has two main features:

1. ChatCSV: Allows users to chat with a CSV file and generate visualizations.
2. ChatPDF: Allows users to chat with a PDF file and retrieve information from it.

## Prerequisites

To run this application, you will need the following:

* Python 3.8 or later
* Streamlit
* PandasAI
* Pandas
* Matplotlib
* PyPDF2
* Langchain
* Your OpenAI API key

## Installation

1. Clone the repository.
2. Install the required packages.
3. Create a `.env` file and add your OpenAI API key (this is very important, don't forget it)
4. Run the application.

## Usage

To run the application, simply run the following command:

```
streamlit run app.py
```

The application will then be available at http://localhost:8501.

## Features

### ChatCSV

The ChatCSV feature allows users to chat with a CSV file and generate visualizations. To use this feature, simply upload a CSV file and start typing your questions. The application will use PandasAI to generate visualizations based on your questions.

### ChatPDF

The ChatPDF feature allows users to chat with a PDF file and retrieve information from it. To use this feature, simply upload a PDF file and start typing your questions. The application will use Langchain to retrieve information from the PDF file.

## Conclusion

This application is a powerful tool that can help users to interact with their data in a conversational way. The application is easy to use and can be used by both technical and non-technical users.


Chat with CSV and PDF using pandasAI, LLM and OpenAI models
