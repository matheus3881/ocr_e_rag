import easyocr
import pymupdf
from PIL import Image
import io 
import cv2
import numpy as np
import asyncio
import os
import faiss


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser




os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
reader = easyocr.Reader(['pt'])  # idioma português



# Função para verificar o tipo do arquivo
def processar_arquivo(arquivo, nome):
    # Verifica se o arquivo é um PDF ou uma imagem
    if nome.lower().endswith(".pdf"):
        return processar_pdf(arquivo)
    elif nome.lower().endswith((".png", ".jpg", ".jpeg")):
        return  [Image.fromarray(preprocessar_imagem(arquivo))]
    else:
        raise ValueError("Formato de arquivo não suportado. Envie um PDF ou PNG.")
    
# Função para processar PDF
def processar_pdf(pdf_path):
    buffer = io.BytesIO(pdf_path)
    buffer.seek(0)

    doc = pymupdf.open(stream=buffer, filetype="pdf")  # ✅ usa o buffer com stream
    imagens = []
    for pagina in doc:
        pix = pagina.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        imagens.append(img)
    return imagens



# ============================================ pre-processar imagem ==================================================== 
def preprocessar_imagem(entrada_imagem):
    # Converte bytes para um buffer de arquivo
    buffer =io.BytesIO(entrada_imagem)
    buffer.seek(0)

    pil_img = Image.open(buffer).convert('RGB')
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray



# ========================================== realizer OCR =================================================
# Função para realizar o OCR
def realizar_ocr(imagens):
    texto_total = []
    
    for img in imagens:
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif not isinstance(img, np.ndarray):
            raise TypeError("Todos os itens devem ser PIL.Image ou numpy.ndarray")

        result = reader.readtext(img)
        texto = " ".join([r[1] for r in result])
        texto_total.append(texto)
        
    return "\n".join(texto_total)

# ================================== dividir texto =========================================================
def dividir_em_chunks(textos_ocr):
    if isinstance(textos_ocr, str):
        textos_ocr = [textos_ocr]  # garante que seja uma lista de strings
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documentos = text_splitter.create_documents(textos_ocr)
    
    return documentos
    

# =========================================== criar recuperador ==============================================
def criar_faiss(chunks: list[Document]):
    if not isinstance(chunks, list) or not all(isinstance(chunk, Document) for chunk in chunks):
        raise TypeError("chunks deve ser uma lista de objetos Document.")
    
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    
    texts = [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in chunks]
    embeddings = np.array(embedder.embed_documents(texts)).astype("float32")
    
    if len(embeddings) == 0:
        raise ValueError("Nenhum embeddinf foi gerado.")
    
    dimesion = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimesion)
    
    index.add(embeddings)
    
    print("Banco vetorial FAISS criado")
    
    return index, embedder
    



# ==================================== perguntar LLM ========================================================
def configurar_llm():
    model = "gemma3:4b"
    llm = ChatOllama(model=model)
    
    
    prompt_template = """
        Você é um assistente especializado em extrair informações de documentos de contrato social, com foco em precisão literal e consistência.

        Analise o conteúdo OCR extraído abaixo e identifique, com base exclusivamente no que está escrito, os seguintes dados:

        Extraia os seguintes dados da forma mais fiel possível ao texto:

        1. Nome da empresa
        2. Número do NIRE
        3. Nomes das pessoas constantes no documento
        4. CPF de cada pessoa mencionada
        5. Endereço ou domicílio de cada pessoa
        6. Endereço ou local da sede da empresa
        7. Número de protocolo (se houver)
        8. Data do documento (ou da assinatura)

        ⚠️ Somente preencha os campos se as informações estiverem explicitamente presentes no texto.  
        Retorne a resposta de forma estruturada, em tópicos.

        Retorne a resposta estritamente em formato **JSON válido**, usando as chaves definidas acima.



        Context: {context}
        
        Pergunta: {question}
        
        Resposta:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain




def buscar_resposta(index, embedder, documentos, query, chain):
    query_embedding = np.array(embedder.embed_query(query)).astype("float32")
    
    I, D = index.search(np.array([query_embedding]), 3)
    
    retrieved_docs = []
    for i in I[0]:
        idx = int(i)
        if idx < len(documentos):
            retrieved_docs.append(documentos[idx].page_content)
            
    if not retrieved_docs:
        return "não encontrei informações no documento."
    
    context = "\n".join(retrieved_docs)
    
    resposta = chain.invoke({"context": context, "question": query})
    return resposta





# def configurar_rag(chunks, prompt_template):
#     if not isinstance(chunks, list) or not all(isinstance(chunk, Document) for chunk in chunks):
#         raise TypeError("chunks deve ser uma lista de objetos Document.")
#     if not isinstance(prompt_template, str):
#         raise TypeError("prompt_template deve ser uma string.")
    
#     retriever = faiss(chunks)
    
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", prompt_template),
#             ("human", "{input}")
#         ]
#     )
    
#     llm = OllamaLLM(model="gemma3:4b")
#     combine_docs_chain = create_stuff_documents_chain(
#         llm=llm,
#         prompt=prompt
#     )
    
#     rag_chain = create_retrieval_chain(
#         retriever=retriever,
#         combine_docs_chain=combine_docs_chain
#     )
    
#     return rag_chain


# ======================= FUNÇÕES ASSÍNCRONAS ===========================

async def processar_arquivo_async(arquivo, nome):
    return await asyncio.to_thread(processar_arquivo, arquivo, nome)

async def realizar_ocr_async(imagens):
    return await asyncio.to_thread(realizar_ocr, imagens)

# ========================= FLUXO PRINCIPAL =============================

async def fluxo_principal_async(arquivo_bytes: bytes, nome_arquivo: str):
    
    query = "extraia os dados conforme especificado no prompt."
    
    
    imagens = await processar_arquivo_async(arquivo_bytes, nome_arquivo)
    texto = await realizar_ocr_async(imagens)
    chunks = dividir_em_chunks([texto])
    index, embedder = criar_faiss(chunks)
    chain = configurar_llm()  
    
    
    resposta = buscar_resposta(index, embedder, chunks, query, chain)

    print("resposta", resposta)
    return resposta



