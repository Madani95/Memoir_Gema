css = '''
<style>
/* Styles for chat messages */
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    align-items: center;
    font-family: Arial, sans-serif;
}

.chat-message.user {
    justify-content: flex-end;
    background-color: #0084FF;
    color: white;
}

.chat-message.bot {
    justify-content: flex-start;
    background-color: #F8F8F8;
    color: black;
}

/* Styles for messages */
.chat-message .message {
    max-width: 60%;
    padding: 1rem;
    border-radius: 1.5rem;
}
</style>
'''

bot_template = '''
<!-- Bot message template -->
<div class="chat-message bot">
    <div class="message">{{content}}</div>
</div>
'''

user_template = '''
<!-- User message template -->
<div class="chat-message user">
    <div class="message">{{content}}</div>
</div>
'''

dark_theme_css = """
<style>
    body {
        background-color: #2e2e2e;
        color: #ffffff;
    }
    .stTextInput label, .stButton>button, .stFileUploader label, .stForm>div>button {
        color: #ffffff;
    }
    .stTextInput>div>input, .stTextArea>div>textarea {
        background-color: #3f3f3f;
        color: #ffffff;
    }
    .stButton>button, .stForm>div>button {
        background-color: #4CAF50;
    }
    .stSidebar .sidebar-content {
        background-color: #363636;
    }
    .stSidebar .sidebar-content .block-container .simplebar-content div[data-baseweb="select"] {
        color: #ffffff;
    }
    .element-container {
        background-color: transparent !important;
    }
    .block-container {
        background-color: transparent !important;
    }
    .stMarkdown code, .stMarkdown pre {
        background-color: #4a4a4a;
    }
    .reportview-container .main .block-container {
        padding: 1rem;
    }
</style>
"""
