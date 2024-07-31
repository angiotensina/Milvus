import streamlit as st

def embed_documents_to_milvus(index, metadata):
    pass

def main():
    #Cabecera de la app
    st.markdown("<H1 style='text-align: center'> RAG PROGRAM </H1>", unsafe_allow_html=True)
    #st.write('This is a simple Streamlit app.')
    col1, col2 = st.columns(2)
    col1.selectbox('IP server', options=['BERT', 'RoBERTa', 'DistilBERT'])
    col2.selectbox('Port', options=['BERT', 'RoBERTa', 'DistilBERT'])


    # Sidebar
    st.sidebar.markdown("<H1 style='text-align: center'> Panel principal </H1>", unsafe_allow_html=True)
    radioselected=st.sidebar.radio('Selecciona una opción', ['Embeddigs', 'RAG', 'Index'])

    # Main panel
    if radioselected == 'Embeddigs':
        with st.form(key='embeddigs_form', clear_on_submit=True):
            st.write('Embeddings')
            col1a, col2a = st.columns(2)
            index=col1a.text_input('Enter Index')
            metadata=col2a.text_input('Enter metadata')
            
            st.form_submit_button('Create Embeddings', on_click=embed_documents_to_milvus(index, metadata))
            
    elif radioselected == 'RAG':
        with st.form(key='RAG_form', clear_on_submit=True):
            st.write('RAG')
            col1b, col2b = st.columns(2)
            col1b.text_input('Enter Index')
            col2b.text_input('Enter metadata')
            st.text_area('Enter RAG question')
            st.text_area('RAG answer', disabled=False, on_change=None, value='')
            st.form_submit_button('Create RAG')

if __name__ == "__main__":
    main()