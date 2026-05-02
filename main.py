import os
import shutil
import streamlit as st
from openai import OpenAI, APIConnectionError
import httpx
import time
import base64
from PyPDF2 import PdfReader
from datetime import datetime
from icalendar import Calendar
import urllib.parse
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# GitHub公開用にSecretsから取得するように変更（未設定時はデフォルト値を使用）
try:
    YAHOO_APP_ID = st.secrets["YAHOO_APP_ID"]
except Exception:
    YAHOO_APP_ID = "dmVyPTIwMjUwNyZpZD1wZVJpUEo4OFV4Jmhhc2g9TXpJeU5EVTNaVEV4WkRZelltTXdZUQ"

DB_DIR = "faiss_index"

# --- 外部API・ユーティリティ関数の定義（呼び出しより前に配置） ---

# Yahoo! ローカル検索APIで座標を特定
def fetch_coordinates(app_id, query):
    if not app_id or not query:
        return None, None
    try:
        url = "https://map.yahooapis.jp/search/local/V1/localSearch"
        params = {
            "appid": app_id,
            "query": query,
            "output": "json",
            "results": 1
        }
        with httpx.Client(trust_env=False) as h_client:
            resp = h_client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                if "Feature" in data and len(data["Feature"]) > 0:
                    feature = data["Feature"][0]
                    return feature["Geometry"]["Coordinates"], feature["Name"]
    except Exception as e:
        st.error(f"座標取得中にエラーが発生しました: {e}")
    return None, None

# Yahoo天気API連携
def fetch_yahoo_weather(app_id, coordinates):
    if not app_id:
        return "Yahoo App IDが設定されていないため、天気情報を取得できません。"
    try:
        if ',' not in coordinates:
            return "座標の形式が正しくありません。 (経度,緯度)"
            
        url = "https://map.yahooapis.jp/weather/V1/place"
        params = {"appid": app_id, "coordinates": coordinates, "output": "json"}
        with httpx.Client(trust_env=False) as h_client:
            resp = h_client.get(url, params=params)
            if resp.status_code != 200:
                return f"天気情報の取得に失敗しました (HTTP {resp.status_code})"
            
            data = resp.json()
            if 'Feature' not in data or not data['Feature']:
                return "指定された座標の気象データが見つかりませんでした。"
            
            place_name = data['Feature'][0].get('Name', '指定地点')
            weather_list = data['Feature'][0].get('Property', {}).get('WeatherList', {}).get('Weather', [])
            
            lon, lat = coordinates.split(',')
            map_url = f"https://map.yahoo.co.jp/place?lat={lat}&lon={lon}&zoom=15"
            
            res_text = f"【設定された場所の情報】\n地点名: {place_name}\nYahoo!マップ: {map_url}\n\n【天気データ】\n"
            for w in weather_list:
                dt = w['Date']
                time_f = f"{dt[8:10]}:{dt[10:12]}"
                w_type = "観測値" if w['Type'] == 'observation' else "予測値"
                res_text += f"- {time_f} ({w_type}): 降水強度 {w['Rainfall']} mm/h\n"
            return res_text
    except Exception as e:
        return f"天気情報の取得中にエラーが発生しました: {str(e)}"

# Yahoo!路線情報から経路詳細を取得
def fetch_transit_data(from_st, to_st):
    from_st = from_st.strip()
    to_st = to_st.strip()
    if not from_st or not to_st:
        return "出発駅と到着駅を両方設定してください。"
    try:
        now = datetime.now()
        base_url = "https://transit.yahoo.co.jp/search/result"
        params = {
            "from": from_st, "to": to_st,
            "y": now.year, "m": f"{now.month:02d}", "d": f"{now.day:02d}",
            "hh": f"{now.hour:02d}", "m1": now.minute // 10, "m2": now.minute % 10,
            "type": 1
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        with httpx.Client(trust_env=False, follow_redirects=True) as h_client:
            resp = h_client.get(base_url, params=params, headers=headers)
            if resp.status_code != 200: return f"経路情報の取得に失敗しました (HTTP {resp.status_code})"
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # エラーメッセージの取得
            error_div = soup.find("div", class_="alertSearch")
            if error_div:
                return f"検索エラー: {error_div.get_text(strip=True)}"
            
            # 候補選択画面が表示された場合
            if "searchSelect" in resp.text:
                soup_sel = BeautifulSoup(resp.text, "html.parser")
                candidates = [td.get_text(strip=True) for td in soup_sel.find_all("td", class_="station")]
                cand_str = "、".join(candidates[:3])
                return f"「{from_st}」または「{to_st}」に複数の候補（{cand_str}など）があります。正確な駅名を入力してください。"

            route_summary = soup.find("div", class_="routeSummary")
            if not route_summary: return "該当する経路が見つかりませんでした。駅名が正しいか確認してください。"

            time_info = route_summary.find("span", class_="time").get_text(strip=True) if route_summary.find("span", class_="time") else "不明"
            fare_info = route_summary.find("li", class_="fare").get_text(strip=True) if route_summary.find("li", class_="fare") else "不明"
            transfer_info = route_summary.find("li", class_="transfer").get_text(strip=True) if route_summary.find("li", class_="transfer") else "不明"
            
            return f"【{from_st} から {to_st} への経路】\n- 時間: {time_info}\n- 運賃: {fare_info}\n- 乗換: {transfer_info}\n- 詳細: {resp.url}"
    except Exception as e:
        return f"経路検索エラー: {str(e)}"

# 運行情報の取得
def fetch_operation_status(line_name):
    line_name = line_name.strip()
    if not line_name: return ""
    try:
        search_url = f"https://transit.yahoo.co.jp/diainfo/search?q={urllib.parse.quote(line_name)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        with httpx.Client(trust_env=False, follow_redirects=True) as h_client:
            resp = h_client.get(search_url, headers=headers)
            soup = BeautifulSoup(resp.text, "html.parser")
            result_table = soup.find("div", id="mdSearchLineResult")
            rows = result_table.find_all("tr") if result_table else []
            if len(rows) > 1:
                cols = rows[1].find_all("td")
                return f"【鉄道運行情報】\n対象: {cols[0].text}\n状態: {cols[1].text}\n詳細: {resp.url}\n"
            return f"「{line_name}」の運行情報を特定できませんでした。"
    except Exception as e:
        return f"運行情報取得エラー: {str(e)}"

# 週間予報 (Open-Meteo)
def fetch_weekly_forecast(coordinates):
    if not coordinates or ',' not in coordinates: return ""
    try:
        lon, lat = coordinates.split(',')
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "daily": "weather_code,temperature_2m_max,temperature_2m_min", "timezone": "Asia/Tokyo"}
        with httpx.Client(trust_env=False) as h_client:
            resp = h_client.get(url, params=params)
            if resp.status_code == 200:
                d = resp.json().get("daily", {})
                res = "【週間予報】\n"
                for i in range(len(d.get("time", []))):
                    res += f"- {d['time'][i]}: {d['temperature_2m_min'][i]}~{d['temperature_2m_max'][i]}℃\n"
                return res
            else:
                return ""
    except Exception:
        # 予報取得に失敗した場合は空文字を返す
        return ""
    return ""

# RAGエンジン: 文書をベクトル化
def build_vector_store(files, base_url, model_name="local-model"):
    all_text = ""
    for f in files:
        if f.type == "application/pdf":
            reader = PdfReader(f)
            all_text += f"\n[File: {f.name}]\n" + "\n".join([p.extract_text() for p in reader.pages])
        elif f.type == "text/plain":
            all_text += f"\n[File: {f.name}]\n" + f.read().decode("utf-8")
        elif f.type == "text/calendar" or f.name.endswith(".ics"):
            if Calendar is None:
                st.error(f"icalendar ライブラリが未インストールのため {f.name} をスキップしました。")
                continue
            try:
                # カレンダーファイルの解析
                cal = Calendar.from_ical(f.read())
                all_text += f"\n[Calendar File: {f.name}]\n"
                for component in cal.walk():
                    if component.name == "VEVENT":
                        summary = component.get('summary')
                        start = component.get('dtstart').dt if component.get('dtstart') else "不明"
                        end = component.get('dtend').dt if component.get('dtend') else "不明"
                        loc = component.get('location', '未設定')
                        all_text += f"- 予定: {summary}, 開始: {start}, 終了: {end}, 場所: {loc}\n"
            except Exception as e:
                st.error(f"カレンダーの解析に失敗しました ({f.name}): {e}")
        f.seek(0)
    
    if not all_text.strip(): return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(all_text)
    
    embeddings = OpenAIEmbeddings(base_url=base_url, api_key="not-needed", model=model_name)
    
    if os.path.exists(DB_DIR):
        vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
        vector_db.add_texts(chunks)
    else:
        vector_db = FAISS.from_texts(chunks, embeddings)
    
    vector_db.save_local(DB_DIR)
    return vector_db

# 1. LM Studioへの接続設定
@st.cache_resource
def get_openai_client(base_url):
    # システムのプロキシ設定を無視するように trust_env=False を設定
    # 接続のタイムアウトを 60秒に延長
    http_client = httpx.Client(trust_env=False, timeout=60.0)
    return OpenAI(
        base_url=base_url,
        api_key="not-needed",
        http_client=http_client
    )

st.set_page_config(page_title="LM Studio Chat", layout="centered")
st.title("🤖 LM Studio Chat")

# サイドバーでの設定
with st.sidebar:
    st.header("Settings")
    
    # LM StudioのURL設定とクライアントの初期化を先に行う
    # クラウドからの場合は Ngrok 等の公開URLを入力する必要があります
    default_url = "http://127.0.0.1:1234/v1"
    lm_url = st.text_input("LM Studio URL / API Endpoint", value=default_url)
    
    # 外部アクセス（クラウド）かどうかの簡易判定と警告
    is_running_on_cloud = os.getenv("STREAMLIT_SERVER_PORT") is None or "streamlit.app" in st.get_option("browser.serverAddress")
    if is_running_on_cloud and ("127.0.0.1" in lm_url or "localhost" in lm_url):
        st.error("⚠️ 接続エラーの原因: クラウド環境からローカルの '127.0.0.1' には接続できません。")
        st.info("💡 ヒント: ngrok 等でローカルポートを公開し、その URL を入力してください。")

    client = get_openai_client(lm_url)

    # --- 機能拡張: ファイルアップローダー ---
    st.subheader("📁 Upload Files")
    uploaded_files = st.file_uploader(
        "画像、PDF、テキスト、カレンダー(ics)をアップロード", 
        type=["png", "jpg", "jpeg", "pdf", "txt", "ics"], 
        accept_multiple_files=True
    )
    
    # 接続確認とモデル一覧の取得
    st.subheader("🤖 Model Selection")
    embedding_model = "local-model" # デフォルト値
    if lm_url:
        # ngrok を使用している場合、/v1 忘れを警告する
        if "ngrok-free" in lm_url and not lm_url.endswith("/v1"):
            st.warning("⚠️ ngrok URL の末尾に '/v1' を追加してください。")

        try:
            models = client.models.list()
            model_list = [m.id for m in models.data] if models.data else []
            if model_list:
                selected_model = st.selectbox("Select Model", model_list)
                # Embedding用モデルが別にある場合は選択できるようにするか、ロード中のものを推測
                embedding_model = st.selectbox("Select Embedding Model", model_list, index=0)
                st.success("LM Studio に接続中")
            else:
                st.warning("モデルが見つかりません。LM Studioでロードしてください。")
                selected_model = "local-model"
        except Exception as e:
            st.error("LM Studio に接続できません。")
            if "127.0.0.1" in lm_url or "localhost" in lm_url:
                st.info("💡 ローカルのアドレスを指定していますが、サーバーがクラウド上にある可能性があります。ngrok 等を使用するか、ローカルで `streamlit run` を実行してください。")
            else:
                st.info(f"詳細エラー: {e}")
            selected_model = "local-model"
    else:
        selected_model = "local-model"

    # --- 機能拡張: 音声入力 ---
    st.subheader("🎤 Voice Input")
    audio_value = st.audio_input("音声を録音してテキスト化")
    if audio_value:
        try:
            # LM StudioがWhisper APIをサポートしている場合の例
            transcript = client.audio.transcriptions.create(model="whisper", file=audio_value)
            st.session_state.voice_text = transcript.text
        except Exception as e:
            st.error(f"音声認識エラー: {e}")

    st.divider()
    
    # --- OCR設定 ---
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        ocr_mode = st.checkbox("OCRモード", help="画像からのテキスト抽出に特化します")
    with col_opt2:
        use_rag = st.checkbox("RAGモード", help="大規模な文書から関連箇所を検索して回答します", value=True)
    
    if use_rag:
        st.info(f"📂 蓄積場所: `{DB_DIR}/`")
        if st.button("🗑️ 蓄積データをクリア"):
            if os.path.exists(DB_DIR):
                shutil.rmtree(DB_DIR)
            if "vector_db" in st.session_state:
                st.session_state.vector_db = None
            st.success("蓄積データを削除しました")

    st.divider()
    st.subheader("☀️ Weather Settings")
    use_weather = st.checkbox("Yahoo天気連携", help="最新の降水情報を取得して回答に反映させます")

    # エリア名から座標を特定する機能
    search_area = st.text_input("エリア名で検索", placeholder="例: 渋谷駅, 大阪市北区")
    if st.button("エリアを特定 (Yahoo!マップ)"):
        if search_area:
            found_coords, found_name = fetch_coordinates(YAHOO_APP_ID, search_area)
            if found_coords:
                st.session_state.target_coords = found_coords
                st.success(f"📍 {found_name} の座標をセットしました")
                st.rerun()
            else:
                st.error("場所が見つかりませんでした")

    target_coords = st.text_input("座標 (経度,緯度)", value=st.session_state.get("target_coords", "139.50372,35.60543"), help="例: 139.50372,35.60543")
    st.session_state.target_coords = target_coords

    st.divider()
    st.subheader("🚃 Transit Settings")
    use_transit = st.checkbox("乗換案内連携", help="目的地へのルート検索リンクを生成できるようにします")
    home_station = st.text_input("マイホーム駅 (出発地)", value=st.session_state.get("home_station", "東京"), help="例: 東京, 渋谷, 横浜")
    dest_station = st.text_input("目的地 (到着地)", value=st.session_state.get("dest_station", ""), placeholder="例: 新宿")
    
    st.caption("🚄 運行情報チェック")
    check_line = st.text_input("運行情報を調べる路線名", placeholder="例: 山手線, 中央線")
    
    st.session_state.dest_station = dest_station
    st.session_state.home_station = home_station

    default_sys_prompt = "あなたは親切で優秀なアシスタントです。"
    if ocr_mode:
        default_sys_prompt = "あなたは高度なOCR専門家です。提供された画像からテキストを正確に、構造を保ったまま抽出してください。解説や挨拶は省き、抽出結果のみを出力してください。"
    
    system_prompt = st.text_area("System Prompt", default_sys_prompt, help="AIの役割を設定します")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    # --- 蓄積状況の可視化 ---
    st.divider()
    if "messages" in st.session_state:
        # 簡易的な文字数計算
        history_chars = sum(len(str(m["content"])) for m in st.session_state.messages)
        st.caption(f"現在の累積コンテキスト量: 約 {history_chars} 文字")
        if history_chars > 20000:
            st.warning("⚠️ 蓄積量が多くなっています。動作が不安定になる可能性があります。")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("更新"):
            st.rerun()
    with col2:
        if st.button("クリア"):
            st.session_state.messages = []
            if "voice_text" in st.session_state:
                del st.session_state.voice_text
            st.rerun()

# 2. セッション状態（チャット履歴）の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 音声認識結果がある場合は入力欄のデフォルト値として使うための処理
input_label = "メッセージを入力してください"

# RAG用インデックスの構築
if use_rag and uploaded_files:
    file_ids = "".join([f.name + str(f.size) for f in uploaded_files])
    if "last_file_ids" not in st.session_state or st.session_state.last_file_ids != file_ids:
        with st.spinner("文書をインデックス中..."):
            st.session_state.vector_db = build_vector_store(uploaded_files, lm_url, model_name=embedding_model)
            st.session_state.last_file_ids = file_ids
            if st.session_state.vector_db:
                st.success("データを蓄積しました")

# アプリ起動時やRAG切り替え時にディスクからロード
if use_rag and st.session_state.get("vector_db") is None:
    if os.path.exists(DB_DIR):
        try:
            embeddings = OpenAIEmbeddings(
                base_url=lm_url, 
                api_key="not-needed", 
                model=embedding_model,
            )
            st.session_state.vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"蓄積データの読み込み失敗: {e}")

# 3. 履歴の表示 (最新のStreamlit chat UIを使用)
# システムメッセージはUIに表示せず、背後で管理します
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item["type"] == "text":
                    st.markdown(item["text"])
                elif item["type"] == "image_url":
                    st.image(item["image_url"]["url"])
        else:
            st.markdown(msg["content"])

# 4. ユーザー入力とAIの応答処理
if prompt := st.chat_input(input_label if not ocr_mode else "OCRの指示を入力（例：全文抽出して）"):
    # --- ファイル解析ロジック ---
    
    # OCRモード時のプロンプト調整
    display_prompt = prompt
    if ocr_mode and uploaded_files:
        prompt = f"画像内のテキストを抽出してください。指示: {prompt}"

    content_list = [{"type": "text", "text": prompt}]

    # --- RAG検索ロジック ---
    # AIが「今日」や「明日」を正しく判定できるように、現在の日時を常に注入します
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")
    external_context = f"【現在の日時】\n{current_time_str}\n"

    if use_rag and st.session_state.get("vector_db"):
        with st.spinner("知識ベースを検索中..."):
            # 関連する上位3チャンクを取得
            docs = st.session_state.vector_db.similarity_search(prompt, k=3)
            external_context += "\n\n【関連資料からの抜粋】\n" + "\n---\n".join([d.page_content for d in docs])

    # --- Yahoo天気情報取得 ---
    if use_weather and YAHOO_APP_ID:
        with st.spinner("最新の天気を取得中..."):
            # リアルタイムの雨情報 (Yahoo)
            weather_info = fetch_yahoo_weather(YAHOO_APP_ID, target_coords)
            # 1週間の予報 (Open-Meteo)
            weekly_info = fetch_weekly_forecast(target_coords)
            external_context += f"\n\n{weather_info}\n\n{weekly_info}"

    # --- 乗換案内用コンテキストの構築 ---
    if use_transit and dest_station:
        with st.spinner("経路を検索中..."):
            transit_data = fetch_transit_data(home_station, dest_station)
            external_context += f"\n\n{transit_data}"
            
    # --- 運行情報の取得 ---
    if check_line:
        status_info = fetch_operation_status(check_line)
        external_context += f"\n\n{status_info}"

    if external_context:
        # プロンプトに外部情報を注入
        content_list[0]["text"] = f"あなたは最新の外部データ（RAGや天気API）にアクセスしています。以下の情報を「現在の事実」として扱い、回答してください。\n{external_context}\n\n質問: {prompt}"

    # RAGがオフの場合、または画像等の処理
    if uploaded_files and not use_rag:
        for f in uploaded_files:
            if f.type.startswith("image/"):
                # 画像データを読み込み、base64エンコード
                img_bytes = f.read()
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                f.seek(0)  # 後続の st.image(f) で画像を表示するためにポインタを先頭に戻す
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{f.type};base64,{base64_image}"}
                })
            elif f.type == "application/pdf":
                reader = PdfReader(f)
                pdf_text = f"\n[PDF内容: {f.name}]\n" + "\n".join([page.extract_text() for page in reader.pages])
                # 極端に長いPDFによるクラッシュを防ぐための簡易制限（例: 先頭30,000文字）
                if len(pdf_text) > 30000:
                    st.warning(f"PDF '{f.name}' が長すぎるため、一部のみを読み込みます。")
                    pdf_text = pdf_text[:30000] + "...(以下略)"
                content_list[0]["text"] +=  pdf_text
            elif f.type == "text/plain":
                text_content = f"\n[ファイル内容: {f.name}]\n" + f.read().decode("utf-8")
                if len(text_content) > 20000:
                    st.warning(f"ファイル '{f.name}' が長すぎるため、一部のみを読み込みます。")
                    text_content = text_content[:20000] + "...(以下略)"
                content_list[0]["text"] += text_content

    # メッセージの保存
    # 修正: external_context がある場合や、画像がある場合は content_list を、それ以外は加工後のテキストを送信
    if len(content_list) > 1:
        user_msg = {"role": "user", "content": content_list}
    else:
        user_msg = {"role": "user", "content": content_list[0]["text"]}
        
    st.session_state.messages.append(user_msg)
    
    with st.chat_message("user"):
        st.markdown(display_prompt)
        if uploaded_files:
            for f in uploaded_files:
                if "image" in f.type: st.image(f)

    # 音声テキストをリセット
    if "voice_text" in st.session_state:
        del st.session_state.voice_text

    # 送信データのサイズチェック（デバッグ・ユーザーへの通知用）
    total_chars = sum(len(c["text"]) for c in content_list if c["type"] == "text")
    if total_chars > 10000:
        st.info(f"💡 大規模なデータを送信中 (約 {total_chars} 文字)。モデルの制限により正しく処理されない場合があります。")

    # AIの応答を生成（ストリーミング形式）
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        full_response = ""
        
        # APIに送るメッセージを構築 (システムプロンプト + 履歴)
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(st.session_state.messages)

        try:
            start_time = time.time()
            token_count = 0
            status_placeholder.caption("🤔 思考中...")

            # ストリーミングを有効にしてリクエスト
            stream = client.chat.completions.create(
                model=selected_model, 
                messages=api_messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    token_count += 1
                    
                    # 統計情報の計算
                    elapsed_time = time.time() - start_time
                    tps = token_count / elapsed_time if elapsed_time > 0 else 0
                    
                    status_placeholder.caption(f"📊 生成中: {token_count} tokens | ⏱️ {elapsed_time:.1f}s | ⚡ {tps:.1f} tok/s")
                    # 記述中のカーソルを表示
                    message_placeholder.markdown(full_response + "▌")
            
            # 完了後の表示更新
            status_placeholder.caption(f"✅ 完了: {token_count} tokens | {time.time() - start_time:.1f}s")
            message_placeholder.markdown(full_response)
            # 応答を履歴に保存
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except APIConnectionError as e:
            st.error("LM Studio サーバーに接続できませんでした。")
            st.info("💡 対策:\n1. LM Studio の Local Server が ON になっているか確認してください。\n2. クラウド実行中の場合、URL に '127.0.0.1' は使用できません。ngrok 等の公開 URL を入力してください。")
        except Exception as e:
            error_msg = f"通信エラーが発生しました: {str(e)}"
            st.error(error_msg)
