import os
import sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.join(base_path, 'Gradio'))

from pydantic_settings import BaseSettings, SettingsConfigDict

from urllib.parse import urlencode
import requests
from jose import jwt

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

GOOGLE_OAUTH_URLS = {
    "userinfo": "https://www.googleapis.com/oauth2/v1/userinfo"
}
#
@app.get("/")
async def index():
    return "hello!!!"

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

@app.get("/google/login")
async def google_login():
    """
    https://accounts.google.com/o/oauth2/auth?
        response_type=code&
        redirect_uri=http://localhost:8000/auth/callback&
        scope=https://mail.google.com/&
        client_id=1001574722416-717v9onq2tumlh47o6fmqog4kotcchve.apps.googleusercontent.com&
        access_type=offline
    """
    base_url = "https://accounts.google.com/o/oauth2/auth"
    params = {
        "response_type": "code",
        "redirect_uri":"http://localhost:8000/google/auth_callback",
        "scope":"https://mail.google.com/",
        "client_id":"1001574722416-717v9onq2tumlh47o6fmqog4kotcchve.apps.googleusercontent.com",
        "access_type":"offline",
    }
    url = f"{base_url}?{urlencode(params)}"
    print(url)
    return RedirectResponse(url=url)
    # return url


@app.get("/google/auth_callback")
async def google_auth_callback(request: Request):

    # print(request)
    code = request.query_params.get("code")
    print('code:', code)
    scope = request.query_params.get("scope")
    print('scope:', scope)

    base_url = "https://oauth2.googleapis.com/token"

    params = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id":"1001574722416-717v9onq2tumlh47o6fmqog4kotcchve.apps.googleusercontent.com",
        "client_secret": "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",
        "redirect_uri": "http://localhost:8000/google/auth_callback"
    }

    url = f"{base_url}?{urlencode(params)}"

    res = requests.post(url)
    print(res.json())
    token = res.json()["access_token"]
    print('token:', token)
    refresh_url = "https://oauth2.googleapis.com/token"
    userinfo_url = "https://www.googleapis.com/oauth2/v1/userinfo"

    res_user = requests.get(userinfo_url, headers={
        "Authorization": f"Bearer {token}"
    })
    print(res_user)

    print(res_user.json())

    return "hello"

@app.post('/google/verify')
async def google_verify(request: Request):
    body = await request.json()
    print('body:', body)

    # code = request.query_params.get('code')
    code = body['code']
    print('code:', code)
    
    base_url = "https://oauth2.googleapis.com/token"

    params = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id":"1001574722416-717v9onq2tumlh47o6fmqog4kotcchve.apps.googleusercontent.com",
        "client_secret": "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",
        # "redirect_uri": "http://localhost:8000/google/auth_callback",
        "redirect_uri": "postmessage",
    }

    url = f"{base_url}?{urlencode(params)}"

    res = requests.post(url)
    print(res.json())
    token = res.json()["access_token"]
    # print('token:', token)

    # decoded = jwt.decode(token, options={"verify_signature": False})

    # data = {
    #     "msg": "verify ok:{body}",
    #     "user": decoded
    # }
    # return data

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get("https://www.googleapis.com/oauth2/v3/userinfo", headers=headers)

    print(response)
    
    user = response.json()
    print(user)
    """
    {
        'sub': '117364235394845038392', 
        'name': 'link yang', 
        'given_name': 'link', 
        'family_name': 'yang', 
        'picture': 'https://lh3.googleusercontent.com/a/ACg8ocJDiwZ96GAJI8wZThpriTuMx3Z9xMkROm6DDqq3EGkZluHzzQ=s96-c', 
        'email': 'link.yang2021@gmail.com', 
        'email_verified': True
    }
    """

    return f"verify ok:{body}"




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
"""
    <script src="https://accounts.google.com/gsi/client" onload="console.log('TODO: add onload function')">
    <scirpt>
        if (google) {{
            const client = google.accounts.oauth2.initCodeClient({{
                client_id: '{GOOGLE_CLIENT_ID}',
                scope: 'openid email profile',
                ux_mode: 'popup',
                callback: (response) => {{
                    const xhr = new XMLHttpRequest();
                    xhr.open('POST', 'http://localhost:8000/google/verify', true);
                    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
                    // Set custom header for CRSF
                    xhr.setRquestHeader('X-Requested-With', 'XmlHttpRequest');
                    xhr.onload = function() {{
                        console.log('Auth code reqsponse: ' + xhr.responseText);
                    }}
                    xhr.send('code='+response.code);
                }}
            }});
        }}
    </script>
    """
head = """
    <script src="https://accounts.google.com/gsi/client" onload="console.log('TODO: add onload function')">
    //<script src="https://accounts.google.com/gsi/client"></script>
"""
js = f"""
()=>{{
        console.log("hello:");
        console.log(google);
        const verify_url = "http://localhost:8000/google/verify";
        let client;
        if (google) {{
            client = google.accounts.oauth2.initCodeClient({{
                client_id: '{GOOGLE_CLIENT_ID}',
                scope: 'openid email profile',
                ux_mode: 'popup',
                callback: (response) => {{
                    const code = response.code;
                    alert("code:"+code);
                    console.log('code:', code);

                    fetch(verify_url, {{
                        method: "POST",
                        headers: {{
                            "Content-Type": "application/json"
                        }},
                        body: JSON.stringify({{code:code}})
                    }})
                        .then(res => {{
                            console.log('res:', res)
                        }})
                    const xhr = new XMLHttpRequest();
                    xhr.open('POST', 'http://localhost:8000/google/verify', true);
                    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
                    // Set custom header for CRSF
                    xhr.setRquestHeader('X-Requested-With', 'XmlHttpRequest');
                    xhr.onload = function() {{
                        console.log('Auth code reqsponse: ' + xhr.responseText);
                    }}
                    xhr.send('code='+response.code);
                }}
            }});
            console.log('client:', client);
        }} else {{

        }}
        
        function handleClickLoginBtn() {{
            console.log("click login");
            if (client) {{
                console.log(client);
                client.requestCode();
            }}
        }}
        //document.addEventListener("DOMContentLoaded", function () {{
        const btn = document.getElementById("login-btn");
        console.log(btn);
        if (btn) {{
            // btn.onclick = handleClickLoginBtn;
            btn.addEventListener("click", handleClickLoginBtn);
            // btn.addEventListener("click", function(){{
            //    handleClickLoginBtn();
            //}});
        }}
        //}});
}}
"""

def build_ui():
    with gr.Blocks(
        title="Gradio + 登录状态",
        css="""
        #user-bar {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 0.5rem;
            margin-left: auto;
        }
        #user-avatar img {
            border-radius: 50%;
            width: 36px;
            height: 36px;
        }
        #login-btn, #logout-btn {
            height: 36px;
        }
        """,
        head=head,
        js=js,
    ) as demo:
        

        with gr.Row():
            gr.Markdown("## 🧠 你的 AI 应用")
            
            with gr.Column(elem_id="user-bar"):
                avatar = gr.Image(visible=False, show_label=False, elem_id="user-avatar")
                email = gr.Markdown(visible=False, elem_id="user-email")
                login_btn = gr.Button("🔐 登录", visible=True, elem_id="login-btn")
                logout_btn = gr.Button("🚪 登出", visible=False, elem_id="logout-btn")

        # 主体内容
        gr.Markdown("### 🎯 功能模块")
        name_input = gr.Textbox(label="名字")
        greet_output = gr.Textbox(label="问候语")
        greet_btn = gr.Button("打招呼")

        def greet(name): return f"你好，{name}!"
        greet_btn.click(fn=greet, inputs=name_input, outputs=greet_output)

        # 登录信息加载函数
        def load_user():
            import requests
            try:
                res = requests.get("http://localhost:8000/me")
                data = res.json()
                if data.get("logged_in"):
                    return (
                        gr.update(value=data["picture"], visible=True),
                        gr.update(value=f"👋 {data['email']}", visible=True),
                        gr.update(visible=False),  # login_btn
                        gr.update(visible=True),   # logout_btn
                    )
            except:
                pass
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        # def go_login(): return gr.redirect("/login")
        # def go_logout(): return gr.redirect("/logout")

        demo.load(fn=load_user, inputs=[], outputs=[avatar, email, login_btn, logout_btn])
        # login_btn.click(fn=go_login)
        # logout_btn.click(fn=go_logout)

    return demo

demo_my = build_ui()

# Gradio in FastAPI, dependency
# app = gr.mount_gradio_app(app, demo, path="/gradio", auth_dependency=auth_dependency)
app = gr.mount_gradio_app(app, demo_my, path="/gradio")

