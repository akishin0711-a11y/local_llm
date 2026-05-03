import os
import shutil
import streamlit as st
from openai import OpenAI, APIConnectionError
import httpx
import time
import base64
from PyPDF2 import PdfReader
from datetime import datetime
from zoneinfo import ZoneInfo
from icalendar import Calendar
import urllib.parse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- 代替Embeddingsクラス ---
class LocalSentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        """文書の埋め込みを生成"""
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text):
        """クエリの埋め込みを生成"""
        return self.model.encode([text], convert_to_numpy=True).tolist()[0]
    
    def __call__(self, text):
        """呼び出し可能にする"""
        return self.embed_query(text)

# --- RAG用Embeddings取得関数 ---
def get_embeddings_for_rag(base_url, model_name="local-model"):
    """RAG用のembeddingsを取得（ローカルのsentence-transformersを使用）"""
    try:
        local_embeddings = LocalSentenceTransformerEmbeddings()
        test_result = local_embeddings.embed_query("test")
        st.info(f"✅ ローカルembeddings使用: {len(test_result)}次元")
        return local_embeddings
    except Exception as local_e:
        st.error(f"❌ ローカルembeddingsも利用できない: {local_e}")
        raise Exception("embeddingsが利用できません")

# GitHub公開用にSecretsから取得するように変更（未設定時はデフォルト値を使用）
try:
    YAHOO_APP_ID = st.secrets["YAHOO_APP_ID"]
except Exception:
    YAHOO_APP_ID = "dmVyPTIwMjUwNyZpZD1wZVJpUEo4OFV4Jmhhc2g9TXpJeU5EVTNaVEV4WkRZelltTXdZUQ"

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "faiss_index"))

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
def get_tokyo_now():
    return datetime.now(ZoneInfo("Asia/Tokyo"))


def fetch_transit_data(from_st, to_st):
    from_st = from_st.strip()
    to_st = to_st.strip()
    if not from_st or not to_st:
        return "出発駅と到着駅を両方設定してください。"
    try:
        now = get_tokyo_now()
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
def register_new_chunks(chunks):
    metadata = load_rag_metadata()
    for chunk in chunks:
        chunk_id = f"chunk_{abs(hash(chunk)) % 1000000}"
        if chunk_id not in metadata["chunks"]:
            metadata["chunks"][chunk_id] = {"count": 0, "last_used": None}
    save_rag_metadata(metadata)


def remove_invalid_faiss_index():
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
        except Exception:
            pass


def build_vector_store(files, base_url, model_name="local-model"):
    all_text = ""
    os.makedirs(DB_DIR, exist_ok=True)
    for f in files:
        if f.type == "application/pdf":
            reader = PdfReader(f)
            page_texts = [p.extract_text() or "" for p in reader.pages]
            all_text += f"\n[File: {f.name}]\n" + "\n".join(page_texts)
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
    
    if not all_text.strip():
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(all_text)
    
    embeddings = get_embeddings_for_rag(base_url, model_name)
    
    try:
        if os.path.exists(DB_DIR):
            try:
                vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
                vector_db.add_texts(chunks)
            except Exception:
                remove_invalid_faiss_index()
                vector_db = FAISS.from_texts(chunks, embeddings)
        else:
            vector_db = FAISS.from_texts(chunks, embeddings)
        vector_db.save_local(DB_DIR)
        register_new_chunks(chunks)
        return vector_db
    except Exception as e:
        st.error(f"RAGインデックスの保存に失敗しました: {e}")
        return None

def add_chat_history_to_rag(base_url, model_name="local-model"):
    """チャット履歴をRAGインデックスに追加"""
    try:
        if not st.session_state.get("messages"):
            st.warning("⚠️ チャット履歴がありません")
            return False
        
        # チャット履歴をテキスト化
        history_text = ""
        for msg in st.session_state.messages:
            role = "ユーザー" if msg["role"] == "user" else "AI"
            if isinstance(msg["content"], list):
                content = ""
                for item in msg["content"]:
                    if item["type"] == "text":
                        content += item["text"]
                    elif item["type"] == "image_url":
                        content += "[画像が添付されました]"
            else:
                content = msg["content"]
            history_text += f"\n[{role}]\n{content}\n"
        
        if not history_text.strip():
            st.warning("⚠️ チャット履歴が空です")
            return False
        
        st.info(f"📝 チャット履歴を処理中... ({len(history_text)} 文字)")
        
        # テキストをチャンク化
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(history_text)
        
        # デバッグ: chunksの内容を確認
        st.info(f"📊 {len(chunks)} 個のチャンクに分割しました")
        if chunks:
            st.info(f"最初のチャンク: {chunks[0][:200]}...")
        else:
            st.error("❌ チャンクが空です")
            return False
        
        # 空のチャンクをフィルタリング
        chunks = [chunk for chunk in chunks if chunk.strip()]
        if not chunks:
            st.warning("⚠️ 有効なチャンクがありません")
            return False
        
        st.info(f"📊 有効なチャンク数: {len(chunks)}")
        
        # RAGインデックスに追加
        embeddings = get_embeddings_for_rag(base_url, model_name)
        
        if os.path.exists(DB_DIR):
            vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
            vector_db.add_texts(chunks)
            st.info("✅ 既存のRAGインデックスに追加しました")
        else:
            # デバッグ: from_textsの前にchunksを確認
            st.info(f"🔍 新しいインデックス作成: chunksタイプ={type(chunks)}, 長さ={len(chunks)}")
            for i, chunk in enumerate(chunks[:3]):  # 最初の3つだけ表示
                st.info(f"  チャンク{i}: タイプ={type(chunk)}, 長さ={len(chunk) if isinstance(chunk, str) else 'N/A'}")
                if isinstance(chunk, str):
                    st.info(f"    内容: {chunk[:100]}...")
            
            vector_db = FAISS.from_texts(chunks, embeddings)
            st.info("✅ 新しいRAGインデックスを作成しました")
        
        vector_db.save_local(DB_DIR)
        register_new_chunks(chunks)
        st.success(f"✅ チャット履歴をRAGに追加しました ({len(chunks)} チャンク)")
        return True
    except Exception as e:
        st.error(f"❌ チャット履歴のRAG追加に失敗: {type(e).__name__}: {e}")
        import traceback
        st.error(f"詳細: {traceback.format_exc()}")
        return False

# --- RAG メタデータ管理 ---
import json

METADATA_FILE = os.path.join(DB_DIR, "rag_metadata.json")
MAX_DB_SIZE_MB = 10000  # 最大サイズ（MB）

def load_rag_metadata():
    """RAGメタデータを読み込む"""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"chunks": {}}
    return {"chunks": {}}

def save_rag_metadata(metadata):
    """RAGメタデータを保存"""
    os.makedirs(DB_DIR, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def update_chunk_usage(chunk_id):
    """チャンクの使用回数と最終使用時刻を更新"""
    metadata = load_rag_metadata()
    if chunk_id not in metadata["chunks"]:
        metadata["chunks"][chunk_id] = {"count": 0, "last_used": None}
    metadata["chunks"][chunk_id]["count"] += 1
    metadata["chunks"][chunk_id]["last_used"] = get_tokyo_now().isoformat()
    save_rag_metadata(metadata)

def get_db_size_mb():
    """FAISS DBのディスク容量を取得（MB）"""
    if not os.path.exists(DB_DIR):
        return 0
    total_size = 0
    for root, dirs, files in os.walk(DB_DIR):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    return total_size / (1024 * 1024)

def cleanup_low_usage_chunks():
    """使用頻度の低いチャンクを削除して容量管理"""
    current_size = get_db_size_mb()
    if current_size <= MAX_DB_SIZE_MB:
        return False  # 削除不要
    
    metadata = load_rag_metadata()
    if not metadata["chunks"]:
        return False
    
    # 使用回数でソート（少ない順）
    sorted_chunks = sorted(
        metadata["chunks"].items(),
        key=lambda x: (x[1]["count"], x[1]["last_used"] or "")
    )
    
    # 最も使用されていないチャンクから削除候補に
    removed_count = 0
    for chunk_id, info in sorted_chunks:
        if current_size <= MAX_DB_SIZE_MB * 0.8:  # 容量が80%以下なら停止
            break
        # メタデータから削除（実際のFAISS削除は再構築が必要なため省略）
        del metadata["chunks"][chunk_id]
        removed_count += 1
        # ディスク容量を再計算
        current_size = get_db_size_mb()
    
    save_rag_metadata(metadata)
    
    # 容量が超過している場合、FAISSインデックスを再構築
    if get_db_size_mb() > MAX_DB_SIZE_MB:
        st.warning(f"⚠️ RAG容量が制限超過。インデックスを最適化中...")
        rebuild_rag_index()
        return True
    
    return removed_count > 0

def rebuild_rag_index():
    """使用中のメタデータに基づいてFAISSインデックスを再構築"""
    # 新しいディレクトリに再構築し、古いものを置き換え
    metadata = load_rag_metadata()
    if not metadata["chunks"] or not os.path.exists(DB_DIR):
        return
    
    try:
        # 既存の有効なチャンクのみを保持
        shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR, exist_ok=True)
        save_rag_metadata(metadata)
        st.success(f"✅ RAGインデックスを最適化しました")
    except Exception as e:
        st.error(f"❌ インデックス再構築エラー: {e}")

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
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        ocr_mode = st.checkbox("OCRモード", help="画像からのテキスト抽出に特化します")
    with col_opt2:
        use_rag = st.checkbox("RAGモード", help="大規模な文書から関連箇所を検索して回答します", value=True)
    with col_opt3:
        use_image_analysis = st.checkbox("画像分析モード", help="アップロードした画像をチャットに送信して分析します", value=True)
    
    if use_rag:
        st.info(f"📂 蓄積場所: `{DB_DIR}/`")
        
        # RAG容量表示
        db_size = get_db_size_mb()
        size_percent = (db_size / MAX_DB_SIZE_MB) * 100 if MAX_DB_SIZE_MB > 0 else 0
        col_size1, col_size2 = st.columns([3, 1])
        with col_size1:
            st.progress(min(size_percent / 100, 1.0), text=f"容量: {db_size:.1f}MB / {MAX_DB_SIZE_MB}MB")
        with col_size2:
            if st.button("🔄 最適化", key="optimize_rag"):
                if cleanup_low_usage_chunks():
                    st.success("✅ 低使用度チャンクを削除しました")
                    st.rerun()
                else:
                    st.info("ℹ️ 最適化は不要です")
        
        # メタデータ表示
        metadata = load_rag_metadata()
        if metadata["chunks"]:
            st.caption(f"📊 登録チャンク数: {len(metadata['chunks'])}")
            # 使用頻度トップ3を表示
            top_chunks = sorted(
                metadata["chunks"].items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:3]
            if top_chunks:
                with st.expander("📈 使用頻度トップ 3"):
                    for i, (chunk_id, info) in enumerate(top_chunks, 1):
                        st.caption(f"{i}. 使用回数: {info['count']} 回 | 最終使用: {info['last_used'][:10] if info['last_used'] else '未使用'}")
        else:
            st.caption("📊 登録チャンク数: 0（RAGインデックス未作成）")
        
        if st.button("🗑️ 蓄積データをクリア"):
            if os.path.exists(DB_DIR):
                shutil.rmtree(DB_DIR)
            if "vector_db" in st.session_state:
                st.session_state.vector_db = None
            st.success("蓄積データを削除しました")
        
        # RAGインデックス強制再構築
        if st.button("🔄 RAGインデックス再構築", key="rebuild_rag"):
            if uploaded_files:
                with st.spinner("RAGインデックスを再構築中..."):
                    remove_invalid_faiss_index()
                    st.session_state.vector_db = build_vector_store(uploaded_files, lm_url, model_name=embedding_model)
                    if st.session_state.vector_db:
                        st.success("✅ RAGインデックスを再構築しました")
                        st.rerun()
                    else:
                        st.error("❌ RAGインデックスの再構築に失敗しました")
            else:
                st.warning("⚠️ 再構築するにはファイルをアップロードしてください")
        
        # チャット履歴をRAGに追加
        if st.button("📝 チャット履歴をRAGに追加", key="add_history_to_rag"):
            if st.session_state.get("messages"):
                with st.spinner("チャット履歴をRAGに追加中..."):
                    success = add_chat_history_to_rag(lm_url, embedding_model)
                    if success:
                        st.rerun()  # 容量表示を更新
            else:
                st.info("ℹ️ チャット履歴がありません")
        
        # 自動RAG追加設定
        auto_add_history = st.checkbox("自動RAG追加", help="会話が10回を超えると自動で履歴をRAGに追加します", value=False)

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
if uploaded_files and use_image_analysis:
    input_label = "画像分析の指示を入力してください。例: この画像について説明してください。"
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
            if not os.path.exists(os.path.join(DB_DIR, "index.faiss")):
                remove_invalid_faiss_index()
            else:
                embeddings = get_embeddings_for_rag(lm_url, embedding_model)
                st.session_state.vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            remove_invalid_faiss_index()
            st.warning(f"蓄積データの読み込み失敗: {e}。破損したインデックスを削除しました。再度アップロードしてください。")

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
    current_time_str = get_tokyo_now().strftime("%Y-%m-%d %H:%M:%S (%A)")
    external_context = f"【現在の日時】\n{current_time_str}\n"

    if use_rag and st.session_state.get("vector_db"):
        with st.spinner("知識ベースを検索中..."):
            # 関連する上位3チャンクを取得
            docs = st.session_state.vector_db.similarity_search(prompt, k=3)
            external_context += "\n\n【関連資料からの抜粋】\n" + "\n---\n".join([d.page_content for d in docs])
            
            # 使用度を記録
            for idx, doc in enumerate(docs):
                chunk_id = f"chunk_{idx}_{abs(hash(doc.page_content)) % 100000}"
                update_chunk_usage(chunk_id)
            
            # 自動削除チェック
            if cleanup_low_usage_chunks():
                st.info("💡 RAG容量が最適化されました")

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

    # アップロードされたファイルをチャットに含める
    if uploaded_files:
        for f in uploaded_files:
            if f.type.startswith("image/") and use_image_analysis:
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
            
            # 自動RAG追加チェック
            if auto_add_history and len(st.session_state.messages) >= 20:  # ユーザー+AIで10往復
                if not st.session_state.get("history_added_to_rag", False):
                    if add_chat_history_to_rag(lm_url, embedding_model):
                        st.session_state.history_added_to_rag = True
                        st.info("💡 チャット履歴をRAGに自動追加しました")
        except APIConnectionError as e:
            st.error("LM Studio サーバーに接続できませんでした。")
            st.info("💡 対策:\n1. LM Studio の Local Server が ON になっているか確認してください。\n2. クラウド実行中の場合、URL に '127.0.0.1' は使用できません。ngrok 等の公開 URL を入力してください。")
        except Exception as e:
            error_msg = f"通信エラーが発生しました: {str(e)}"
            st.error(error_msg)
