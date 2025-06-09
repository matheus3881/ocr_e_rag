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
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
reader = easyocr.Reader(['pt'])  # idioma português


# Função para verificar o tipo do arquivo
def processar_arquivo(arquivo, nome):
    # Verifica se o arquivo é um PDF ou uma imagem
    if nome.lower().endswith(".pdf"):
        return processar_pdf(arquivo)
    elif nome.lower().endswith((".png", ".jpg", ".jpeg")):
        return [Image.fromarray(preprocessar_imagem(arquivo))]
    else:
        raise ValueError(
            "Formato de arquivo não suportado. Envie um PDF ou PNG.")

# Função para processar PDF


def processar_pdf(pdf_path):
    buffer = io.BytesIO(pdf_path)
    buffer.seek(0)

    # ✅ usa o buffer com stream
    doc = pymupdf.open(stream=buffer, filetype="pdf")
    imagens = []
    for pagina in doc:
        pix = pagina.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        imagens.append(img)
    return imagens


# ============================================ pre-processar imagem ====================================================
def preprocessar_imagem(entrada_imagem):
    # Converte bytes para um buffer de arquivo
    buffer = io.BytesIO(entrada_imagem)
    buffer.seek(0)

    pil_img = Image.open(buffer).convert('RGB')
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


# ========================================== realizer OCR =================================================
# Função para realizar o OCR
def realizar_ocr(imagens):
    documentos = []

    for i, img in enumerate(imagens):
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif not isinstance(img, np.ndarray):
            raise TypeError(
                "Todos os itens devem ser PIL.Image ou numpy.ndarray")

        result = reader.readtext(img)

        # Junta todas as linhas de texto da página
        texto_pagina = "\n".join([r[1] for r in result])

        # Cria um único Document por página
        documento = Document(
            page_content=texto_pagina,
            metadata={"source": f"Página {i + 1}"}
        )

        documentos.append(documento)

    return documentos

# ================================== dividir texto =========================================================


def dividir_em_chunks(textos_ocr):
    if isinstance(textos_ocr, dict):
        textos_ocr = [textos_ocr]  # garante que seja uma lista de strings

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    documentos = []

    for item in textos_ocr:
        chunks = text_splitter.create_documents([item.page_content])
        for chunk in chunks:
            doc = Document(page_content=chunk.page_content,
                           metadata=item.metadata)
            documentos.append(doc)

    return documentos


# =========================================== criar recuperador ==============================================
def criar_faiss(chunks: list[Document]):
    if not isinstance(chunks, list) or not all(isinstance(chunk, Document) for chunk in chunks):
        raise TypeError("chunks deve ser uma lista de objetos Document.")

    embedder = OllamaEmbeddings(model="nomic-embed-text")

    texts = [doc.page_content for doc in chunks]
    embeddings = np.array(embedder.embed_documents(texts)).astype("float32")

    if len(embeddings) == 0:
        raise ValueError("Nenhum embeddinf foi gerado.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("Banco vetorial FAISS criado")

    return index, embedder, chunks


# ==================================== perguntar LLM ========================================================
def configurar_llm():
    model = "gemma3:4b"
    llm = ChatOllama(model=model)

    prompt_template = """
        Você é um assistente especializado em extrair informações de documentos de contrato social, com foco em precisão literal e consistência.

        Analise o conteúdo OCR extraído abaixo e identifique, com base exclusivamente no que está escrito, os seguintes dados:

        Extraia os seguintes dados da forma mais fiel possível ao texto
        
        Retorne a resposta em formato de lista com marcadores, como no exemplo:

        - Nome da empresa: ...
        - Número do NIRE: ...
        - Nomes das pessoas constantes no documento: ...
        - CPF de cada pessoa mencionada: ...
        - Endereço ou domicílio de cada pessoa: ...
        - Endereço ou local da sede da empresa: ...
        - Número de protocolo (se houver): ...
        - Data do documento (ou da assinatura): ...

        Somente preencha os campos se as informações estiverem explicitamente presentes no texto.
        
        Retorne a resposta estritamente em formato de LISTA.



        Context: {context}
        
        Pergunta: {question}
        
        Resposta:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    print("prompt:", prompt)

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("chain:", chain)
    return chain


def buscar_resposta(index, embedder, documentos, query, chain):
    query_embedding = np.array(embedder.embed_query(query)).astype("float32")

    I, D = index.search(np.array([query_embedding]), 3)

    retrieved_docs = []
    fontes = set()
    for i in I[0]:
        idx = int(i)
        if idx < len(documentos):
            doc = documentos[idx]
            retrieved_docs.append(documentos[idx].page_content)

            source = doc.metadata.get("source", "desconhecido")
            fontes.add(source)

    if not retrieved_docs:
        return "não encontrei informações no documento."


    for doc in retrieved_docs:
        context = "\n".join(retrieved_docs)

    resposta = chain.invoke({"context": context, "question": query})
    return {
        "resposta": resposta,
        "fontes": list(fontes)
    }


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
    chunks = dividir_em_chunks(texto)
    index, embedder, chunks = criar_faiss(chunks)
    chain = configurar_llm()

    resposta = buscar_resposta(index, embedder, chunks, query, chain)

    print("resposta", resposta)
    return resposta
