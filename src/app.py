import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from ia import fluxo_principal_async
import traceback


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/extrair/")
async def upload(arquivo: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    
    try:

        conteudo = await arquivo.read()
        file_location = f"uploads/{arquivo.filename}"
        

        # Salva o arquivo no disco
        with open(file_location, "wb") as f:
            f.write(conteudo)

        nome = arquivo.filename
        resposta = await fluxo_principal_async(conteudo, nome)

        return resposta
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"erro": str(e)}, status_code=500)
