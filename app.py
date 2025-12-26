import streamlit as st
import os
from llama_cpp import Llama

# ページ設定
st.set_page_config(page_title="Local GGUF Chat", layout="wide")

# タイトル
st.title("🦙 GGUF Chat in Codespaces")

# ---------------------------------------------------------
# サイドバー: モデルのアップロードと設定
# ---------------------------------------------------------
st.sidebar.header("モデル設定")

# GGUFファイルのアップロード機能
uploaded_file = st.sidebar.file_uploader("GGUFファイルをアップロード", type=["gguf"])

# モデルを保存するディレクトリ
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

selected_model_path = None

if uploaded_file is not None:
    # ファイルをディスクに保存する（llama.cppはファイルパスが必要なため）
    file_path = os.path.join(MODEL_DIR, uploaded_file.name)
    
    # すでに同名ファイルがない場合、または再アップロード時に保存
    if not os.path.exists(file_path):
        with st.sidebar.status("ファイルを保存中..."):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"保存完了: {uploaded_file.name}")
    
    selected_model_path = file_path
    st.sidebar.info(f"使用モデル: {uploaded_file.name}")

# パラメータ設定
n_ctx = st.sidebar.slider("コンテキストサイズ (n_ctx)", 512, 4096, 2048, step=256)
temperature = st.sidebar.slider("Temperature (創造性)", 0.0, 1.0, 0.7)

# モデルロードボタン
if selected_model_path:
    if st.sidebar.button("モデルをロード/リロード"):
        # セッションステートのモデルをクリア
        if "llm" in st.session_state:
            del st.session_state["llm"]
        
        try:
            with st.spinner("モデルをロード中... (CPUでの処理のため時間がかかります)"):
                # モデルの初期化
                st.session_state.llm = Llama(
                    model_path=selected_model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=0, # CPUのみ
                    verbose=False
                )
            st.sidebar.success("モデルロード完了！")
        except Exception as e:
            st.sidebar.error(f"エラーが発生しました: {e}")

# ---------------------------------------------------------
# メインエリア: チャット UI
# ---------------------------------------------------------

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

# 履歴の表示
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# チャット入力
if prompt := st.chat_input("メッセージを入力..."):
    # ユーザーの入力を表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AIの返答生成
    if "llm" in st.session_state:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # ストリーミング生成
            stream = st.session_state.llm.create_chat_completion(
                messages=st.session_state.messages,
                temperature=temperature,
                stream=True
            )
            
            for chunk in stream:
                if "content" in chunk["choices"][0]["delta"]:
                    content = chunk["choices"][0]["delta"]["content"]
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.warning("左側のサイドバーからGGUFファイルをアップロードし、「モデルをロード」ボタンを押してください。")
