import streamlit as st
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_models():
    model_summary = pipeline("summarization", model="facebook/bart-large-cnn")
    model_QA = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return model_summary, model_QA

def generate_chunks(inp_str, max_chunk=500):
    inp_str = inp_str.replace('.','.<eos>')
    inp_str = inp_str.replace('?','?<eos>')
    inp_str= inp_str.replace('!','!<eos>')
    sentences = inp_str.split('<eos>')
    current_chunk = 0
    chunk = []
    for sentence in sentences:
        if len(chunk) == current_chunk +1:
            if len(chunk[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunk[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk +=1
                chunk.append(sentence.split(' '))
        else:
            chunk.append(sentence.split(' '))
    for chunk_id in range(len(chunk)):
        chunk[chunk_id] = ' '.join(chunk[chunk_id])
    return chunk

def main():
    st.title("📝 Text Summarization & Q/A System")
    
    with st.spinner("Loading AI models..."):
        model_summary, model_QA = load_models()
    
    default_text = 'NLP or Natural Language Processing is an AI field focused on computer-human language interaction. It aims to enable machines to understand, interpret, and generate human text meaningfully.Machine translation helps break language barriers between cultures. Virtual assistants like Siri and Alexa use advanced techniques to respond to voice commands. Sentiment analysis examines texts to identify positive or negative opinions.The evolution of large language models like GPT-4 and BERT continues to push boundaries in what machines can achieve with human language. Multimodal AI systems that combine text, audio, and visual processing are creating more comprehensive understanding capabilities. Ethical considerations around bias mitigation and responsible AI deployment are becoming increasingly important as these technologies become more integrated into daily life. The ongoing research in explainable AI aims to make NLP systems more transparent and trustworthy.'
    
    st.header("Enter Your Text")
    input_text = st.text_area(
        "Paste your article below:",
        value=default_text,
        height=200
    )
    
    st.header("Text Summarization")
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                chunks = generate_chunks(input_text)
                res = model_summary(chunks, max_length=100, min_length=10)
                text = ' '.join([summ['summary_text'] for summ in res])
                
                st.success("✅ Summary Generated!")
                st.subheader("Summary:")
                st.write(text)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.header("❓ Question Answering")
    
    questions = [
        "What does NLP aim to enable machines to do?",
        "What are examples of virtual assistants?",
        "What is sentiment analysis?",
        "What are large language models?"
    ]
    cols = st.columns(2)
    for i, q in enumerate(questions):
        with cols[i % 2]:
            if st.button(f"❓ {q}", key=f"btn_{i}"):
                st.session_state.selected_question = q
    custom_question = st.text_input(
        "Or ask your own question:",
        placeholder="Type your question here..."
    )
    
    question_to_ask = custom_question if custom_question else st.session_state.get('selected_question', '')
    
    if question_to_ask:
        with st.spinner("Finding answer..."):
            try:
                answers = model_QA(question=question_to_ask, context=input_text)
                
                st.success("Answer Found!")
                st.subheader("Question:")
                st.write(question_to_ask)
                st.subheader("Answer:")
                st.info(answers['answer'])
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":

    main()
