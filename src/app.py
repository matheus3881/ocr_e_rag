from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from ia import fluxo_principal_async


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/extrair/")
async def upload(arquivo: UploadFile = File(...)):
    try:
        conteudo = await arquivo.read()
        nome = arquivo.filename

        resposta = await fluxo_principal_async(conteudo, nome)
        return {"resultado": resposta}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"erro": str(e)}, status_code=500)


