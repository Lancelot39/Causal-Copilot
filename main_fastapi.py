import os
import sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.join(base_path, 'Gradio'))

from pydantic_settings import BaseSettings, SettingsConfigDict

from fastapi import FastAPI, Request, Depends
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
import gradio as gr

from Gradio.demo import demo

# from dotenv import load_dotenv
# load_dotenv()

class Settings(BaseSettings):
    # google_client_id: str = ""
    GOOGLE_CLIENT_ID: str = ""
    # google_client_secret: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )

settings = Settings()

# FastAPI app
app = FastAPI()

# print(settings)
# print(settings.model_extra)

# OAuth
# GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
# GOOGLE_CLIENT_ID = settings.google_client_id
GOOGLE_CLIENT_ID = settings.GOOGLE_CLIENT_ID
print("client_id:", GOOGLE_CLIENT_ID)
print(settings.GOOGLE_CLIENT_ID)
# GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
# GOOGLE_CLIENT_SECRET = settings.google_client_secret
GOOGLE_CLIENT_SECRET = settings.GOOGLE_CLIENT_SECRET
print("client_secret:", GOOGLE_CLIENT_SECRET)

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


demo_test = gr.Interface(fn=greet, inputs="text", outputs="text")

# Gradio in FastAPI, dependency
# app = gr.mount_gradio_app(app, demo, path="/gradio", auth_dependency=auth_dependency)
app = gr.mount_gradio_app(app, demo, path="/gradio")
