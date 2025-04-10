from fastapi import FastAPI, Request, Depends
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
import gradio as gr

# FastAPI app
app = FastAPI()

# OAuth
GOOGLE_CLIENT_ID = ""
GOOGLE_CLIENT_SECRET = ""

# OAuth register
oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={'scope': 'openid email profile'},
)

# users
logged_in_users = set()

# login
@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for('auth_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

# auth/callback
@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user = await oauth.google.parse_id_token(request, token)
    #
    email = user.get("email")

    #
    logged_in_users.add(email)
    response = RedirectResponse(url="/gradio")
    response.set_cookie("user_email", email)
    return response

# Gradio auth method
async def auth_dependency(request: Request):
    email = request.cookies.get("user_email")
    if email in logged_in_users:
        return email
    return None

# Gradio app
def greet(name):
    return f"Hello {name}!"


demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# Gradio in FastAPI, dependency
app = gr.mount_gradio_app(app, demo, path="/gradio", auth_dependency=auth_dependency)
