import os
import re
import gspread
import pandas as pd
import json
import google.generativeai as genai
import streamlit as st
from PIL import Image
from datetime import datetime
import plotly.express as px
import hashlib
import base64
from io import BytesIO

# --- 1. 設定區 ---
st.set_page_config(page_title="AI 發票記帳助理", page_icon="🔐", layout="wide")
GOOGLE_SHEET_NAME = '我的AI記帳本'
ADMIN_USERNAME = "jerry" # 設定管理員帳號名稱

# --- 2. AI 與 Google 服務核心函式 ---

@st.cache_resource
def get_google_sheet(sheet_name):
    """連線到指定的 Google Sheet"""
    try:
        creds_json = st.secrets["GOOGLE_CREDENTIALS"]
        gc = gspread.service_account_from_dict(creds_json)
        sh = gc.open(sheet_name)
        return sh
    except Exception as e:
        st.error(f"Google Sheet 連線失敗，請檢查 Streamlit Secrets 設定。錯誤訊息: {e}")
        return None

def configure_gemini():
    """設定 Gemini API 金鑰"""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return True
    except KeyError:
        st.error("找不到 Gemini API 金鑰，請確認您已在 .streamlit/secrets.toml 中設定好 GEMINI_API_KEY。")
        return False
    except Exception as e:
        st.error(f"Gemini API 金鑰設定失敗。錯誤訊息: {e}")
        return False

def parse_with_gemini(image_input):
    """使用 Gemini AI 直接解析圖片，同時提取日期和品項。"""
    if not configure_gemini():
        return None
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt_parts = [
        "你是一位頂尖的發票分析師。",
        "請直接分析這張圖片，將其內容解析成一個單一的 JSON 物件。",
        "這個 JSON 物件必須包含兩個鍵: 'invoice_date' 和 'items'。",
        "1. 'invoice_date': 發票上的日期，格式必須是 'YYYY-MM-DD'。如果看到民國年，請轉換成西元年。如果找不到日期，則回傳 null。",
        "2. 'items': 一個 JSON 陣列，包含所有消費品項。",
        "   - 每一筆消費都是一個 JSON 物件，必須包含 '品項', '數量', '類別', '金額' 四個鍵。",
        "   - 『數量』必須是整數，如果圖片中沒有明確數量，預設為 1。",
        "   - 根據品項名稱，自動判斷其「類別」，例如：餐飲食品, 生活用品, 電腦/電子產品, 交通, 其他。",
        "   - 如果品項名稱有多行，請將它們合併成一個字串。",
        "   - 如果遇到金額為 0 的品項，請直接忽略。",
        "請只回傳這個單一的 JSON 物件，不要有其他任何文字說明。",
        "範例格式: {\"invoice_date\": \"2023-03-18\", \"items\": [{\"品項\": \"範例品項\", \"數量\": 1, \"類別\": \"範例類別\", \"金額\": 100}]}"
    ]
    prompt = "\n".join(prompt_parts)
    try:
        response = model.generate_content([prompt, image_input])
        cleaned_response = re.sub(r'```json\n?|```', '', response.text.strip())
        return json.loads(cleaned_response)
    except Exception as e:
        st.error(f"AI 解析時發生錯誤: {e}")
        try:
            st.text_area("AI 原始回傳內容", response.text)
        except:
            pass
        return None

# --- 3. 使用者認證相關函式 (含頭像) ---

def hash_password(password):
    """將密碼進行 SHA-256 加密"""
    return hashlib.sha256(password.encode()).hexdigest()

def crop_to_square(image: Image.Image):
    """將 PIL 圖片從中心裁切成正方形"""
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size)/2
    top = (height - new_size)/2
    right = (width + new_size)/2
    bottom = (height + new_size)/2
    return image.crop((left, top, right, bottom))

def get_users_worksheet(sheet):
    """獲取或建立使用者資料工作表，並確保頭像欄位存在"""
    try:
        worksheet = sheet.worksheet("Users")
        header = worksheet.row_values(1)
        if 'avatar_base64' not in header:
            worksheet.update_cell(1, len(header) + 1, 'avatar_base64')
        return worksheet
    except gspread.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title="Users", rows="100", cols="3")
        worksheet.update('A1:C1', [['username', 'hashed_password', 'avatar_base64']])
        return worksheet

def check_login(username, password):
    """檢查登入資訊"""
    sheet = get_google_sheet(GOOGLE_SHEET_NAME)
    if not sheet: return False
    users_ws = get_users_worksheet(sheet)
    users = users_ws.get_all_records()
    hashed_password_to_check = hash_password(password)
    for user in users:
        # --- 程式碼變更處：使用 .lower() 進行不分大小寫的比對 ---
        if str(user.get('username')).lower() == username.lower() and user.get('hashed_password') == hashed_password_to_check:
            return True, user.get('username') # 回傳資料庫中正確大小寫的名稱
    return False, None

def add_user(username, password, avatar_file):
    """新增使用者，包含頭像"""
    sheet = get_google_sheet(GOOGLE_SHEET_NAME)
    if not sheet:
        st.error("無法連線到資料庫，暫時無法註冊。")
        return False, "資料庫連線失敗"
        
    users_ws = get_users_worksheet(sheet)
    users = users_ws.get_all_records()
    
    # --- 程式碼變更處：強化使用者名稱檢查 ---
    if username.lower() == ADMIN_USERNAME.lower():
        return False, "這個使用者名稱為管理員保留，請選擇其他名稱。"
    if any(str(user.get('username')).lower() == username.lower() for user in users):
        return False, "這個使用者名稱已經被註冊了！"
    
    avatar_base64 = ""
    if avatar_file is not None:
        try:
            img = Image.open(avatar_file)
            img_square = crop_to_square(img)
            img_square.thumbnail((150, 150))
            buffered = BytesIO()
            img_square.save(buffered, format="PNG")
            avatar_base64 = base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            st.warning(f"頭像處理失敗: {e}")

    hashed_password = hash_password(password)
    users_ws.append_row([username, hashed_password, avatar_base64])
    return True, "註冊成功！現在您可以用新帳號登入。"

def update_user(username, new_password=None, new_avatar_file=None):
    """更新使用者資料，包含密碼和頭像"""
    sheet = get_google_sheet(GOOGLE_SHEET_NAME)
    if not sheet:
        return False, "資料庫連線失敗"
    
    users_ws = get_users_worksheet(sheet)
    
    try:
        cell = users_ws.find(username)
    except gspread.CellNotFound:
        return False, "找不到該使用者"

    row_index = cell.row
    
    if new_avatar_file is not None:
        try:
            img = Image.open(new_avatar_file)
            img_square = crop_to_square(img)
            img_square.thumbnail((150, 150))
            buffered = BytesIO()
            img_square.save(buffered, format="PNG")
            avatar_base64 = base64.b64encode(buffered.getvalue()).decode()
            users_ws.update_cell(row_index, 3, avatar_base64)
        except Exception as e:
            return False, f"頭像更新失敗: {e}"
            
    if new_password:
        hashed_password = hash_password(new_password)
        users_ws.update_cell(row_index, 2, hashed_password)
        
    return True, "帳戶資料更新成功！"

def delete_user(username_to_delete):
    """刪除使用者及其所有相關資料"""
    sheet = get_google_sheet(GOOGLE_SHEET_NAME)
    if not sheet:
        return False, "資料庫連線失敗"

    # 刪除使用者帳號
    users_ws = get_users_worksheet(sheet)
    try:
        cell = users_ws.find(username_to_delete)
        users_ws.delete_rows(cell.row)
    except gspread.CellNotFound:
        return False, "在使用者列表中找不到該使用者"
    except Exception as e:
        return False, f"刪除使用者時發生錯誤: {e}"

    # 刪除該使用者的所有消費紀錄
    try:
        data_ws = sheet.worksheet("工作表1")
        all_data = data_ws.get_all_records()
        if all_data:
            df = pd.DataFrame(all_data)
            # 保留不屬於該使用者的資料
            df_remaining = df[df['使用者'] != username_to_delete]
            # 清空工作表並寫回剩餘資料
            data_ws.clear()
            if not df_remaining.empty:
                data_ws.update([df_remaining.columns.values.tolist()] + df_remaining.values.tolist(), 'A1')
            else: # 如果刪除後沒有任何資料了，就只寫入表頭
                data_ws.update([df.columns.values.tolist()], 'A1')

    except gspread.WorksheetNotFound:
        pass
    except Exception as e:
        return False, f"刪除消費紀錄時發生錯誤: {e}"
        
    return True, f"已成功刪除使用者「{username_to_delete}」及其所有資料。"


@st.cache_data(ttl=60)
def get_all_users():
    """獲取所有使用者資料"""
    sheet = get_google_sheet(GOOGLE_SHEET_NAME)
    if not sheet: return []
    users_ws = get_users_worksheet(sheet)
    return users_ws.get_all_records()

# --- 4. 頁面函式 ---
def page_invoice_processing(username):
    st.title(f"🧠 {username} 的 AI 發票辨識")
    st.info("您可以直接用相機拍照，或上傳現有圖片，AI 將自動為您解析。")
    if 'parsed_df' not in st.session_state: st.session_state.parsed_df = None
    if 'uploaded_file_content' not in st.session_state: st.session_state.uploaded_file_content = None
    if 'uploaded_file_name' not in st.session_state: st.session_state.uploaded_file_name = None
    tab1, tab2 = st.tabs(["📷 拍照上傳", "📂 檔案上傳"])
    with tab1: camera_input = st.camera_input("點擊按鈕開啟相機拍攝發票")
    with tab2: file_uploader_input = st.file_uploader("從手機或電腦選擇圖片檔案", type=["png", "jpg", "jpeg"])
    uploaded_file = camera_input or file_uploader_input
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 3])
        with col1:
            if getattr(st.session_state, 'uploaded_file_name', None) != uploaded_file.name:
                st.session_state.parsed_df = None
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.uploaded_file_content = uploaded_file.getvalue()
            image = Image.open(uploaded_file)
            st.image(image, caption="您上傳的圖片", use_container_width=True)
        with col2:
            if st.button("1. 開始辨識", type="primary", use_container_width=True):
                if st.session_state.uploaded_file_content:
                    with st.spinner("AI 正在解析您的發票..."):
                        image_input = Image.open(uploaded_file)
                        parsed_data = parse_with_gemini(image_input)
                        if not parsed_data or not isinstance(parsed_data, dict) or "items" not in parsed_data or not parsed_data.get("items"):
                            st.warning("AI 無法自動解析出任何品項。請手動新增資料。")
                            invoice_date = datetime.now().strftime('%Y-%m-%d')
                            df = pd.DataFrame([{'日期': invoice_date, '品項': '', '數量': 1, '類別': '其他', '金額': 0}])
                        else:
                            parsed_items = parsed_data.get("items", [])
                            invoice_date = parsed_data.get("invoice_date") or datetime.now().strftime('%Y-%m-%d')
                            st.success(f"AI 成功解析出 {len(parsed_items)} 個品項！請在下方表格中校正。")
                            df = pd.DataFrame(parsed_items)
                            df['日期'] = invoice_date
                        st.session_state.parsed_df = df[['日期', '品項', '數量', '類別', '金額']]
            if st.session_state.parsed_df is not None:
                st.subheader("2. 校正辨識結果")
                st.info("您可以直接點擊表格內的儲存格進行修改。")
                df_for_editing = st.session_state.parsed_df.copy()
                df_for_editing['金額'] = pd.to_numeric(df_for_editing['金額'], errors='coerce')
                df_for_editing['數量'] = pd.to_numeric(df_for_editing['數量'], errors='coerce').fillna(1).astype(int)
                edited_df = st.data_editor(df_for_editing, num_rows="dynamic", use_container_width=True, column_config={"金額": st.column_config.NumberColumn("金額 (NT$)", help="請輸入消費金額", format="%d"), "數量": st.column_config.NumberColumn("數量", help="請輸入購買數量", format="%d")})
                if st.button("💾 確認並儲存", type="primary", use_container_width=True):
                    if not edited_df.empty:
                        with st.spinner("正在儲存資料..."):
                            try:
                                final_df_to_save = edited_df.dropna(subset=['金額', '品項', '數量'])
                                final_df_to_save = final_df_to_save[final_df_to_save['品項'].str.strip() != '']
                                final_df_to_save['金額'] = final_df_to_save['金額'].astype(int)
                                final_df_to_save['數量'] = final_df_to_save['數量'].astype(int)
                                final_df_to_save['使用者'] = username
                                if not final_df_to_save.empty:
                                    sheet = get_google_sheet(GOOGLE_SHEET_NAME)
                                    if sheet:
                                        worksheet_daily = sheet.worksheet("工作表1")
                                        header_daily = worksheet_daily.row_values(1)
                                        if not header_daily or '使用者' not in header_daily:
                                            worksheet_daily.clear()
                                            worksheet_daily.update([final_df_to_save.columns.values.tolist()] + final_df_to_save.values.tolist(), 'A1')
                                        else:
                                            worksheet_daily.append_rows(final_df_to_save.values.tolist(), value_input_option='USER_ENTERED')
                                        st.success("資料已成功寫入您的記帳本！")
                                        st.balloons()
                                        st.session_state.parsed_df = None; st.session_state.uploaded_file_name = None; st.session_state.uploaded_file_content = None
                                        st.rerun()
                                else: st.warning("校正後的資料無效或不完整，無法儲存。")
                            except Exception as e: st.error(f"儲存過程中發生錯誤：{e}")
                    else: st.warning("表格中沒有資料，無法儲存。")
    else: st.info("↑ 請從上方選擇拍照或上傳您的發票圖片以開始使用。")

def page_dashboard(username):
    st.title(f"📊 {username} 的消費儀表板")
    @st.cache_data(ttl=600)
    def load_data():
        sheet = get_google_sheet(GOOGLE_SHEET_NAME)
        if sheet:
            try:
                worksheet = sheet.worksheet("工作表1")
                data = worksheet.get_all_records()
                if not data: return pd.DataFrame(), 0
                df = pd.DataFrame(data)
                required_cols = ['日期', '品項', '數量', '金額', '使用者']
                if not all(col in df.columns for col in required_cols):
                    st.warning(f"工作表缺少必要的欄位 ({', '.join(required_cols)})，無法進行分析。")
                    return pd.DataFrame(), 0
                original_rows = len(df)
                df['金額'] = pd.to_numeric(df['金額'], errors='coerce')
                df['數量'] = pd.to_numeric(df['數量'], errors='coerce')
                df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
                df.dropna(subset=required_cols, inplace=True)
                df.drop_duplicates(subset=required_cols, keep='first', inplace=True)
                deduplicated_rows = original_rows - len(df)
                return df, deduplicated_rows
            except gspread.WorksheetNotFound:
                st.warning("找不到名為「工作表1」的分頁，請確認您的 Google Sheet。")
                return pd.DataFrame(), 0
        return pd.DataFrame(), 0
    df_all, removed_count = load_data()
    if df_all.empty:
        st.warning("您的記帳本中尚無有效資料，請先去『發票辨識』頁面上傳資料。")
        return
    df = df_all[df_all['使用者'] == username].copy()
    if df.empty:
        st.info(f"Hi {username}！您的專屬帳本中還沒有資料，快去上傳第一張發票吧！")
        return
    if removed_count > 0: st.success(f"💡 為了數據準確，系統已自動為所有使用者過濾掉 {removed_count} 筆重複的消費紀錄。")
    st.header("🔍 商品搜尋")
    search_term = st.text_input("輸入商品關鍵字來篩選您的消費紀錄：", placeholder="例如：牛奶、咖啡...")
    if search_term:
        df = df[df['品項'].str.contains(search_term, case=False, na=False)]
        if df.empty:
            st.info(f"找不到您包含「{search_term}」的消費紀錄。")
            return
    try:
        df['月份'] = df['日期'].dt.to_period('M').astype(str)
    except Exception as e:
        st.error(f"資料格式錯誤，無法進行分析。錯誤訊息: {e}")
        st.dataframe(df)
        return
    st.header("消費總覽")
    total_expense = df['金額'].sum()
    monthly_total = df.groupby('月份')['金額'].sum()
    monthly_avg = monthly_total.mean() if not monthly_total.empty else 0
    col1, col2 = st.columns(2)
    col1.metric("總支出金額", f"NT$ {int(total_expense):,}")
    col2.metric("平均每月支出", f"NT$ {int(monthly_avg):,}")
    st.subheader("每月消費趨勢")
    monthly_summary = monthly_total.reset_index()
    fig_bar = px.bar(monthly_summary, x='月份', y='金額', title="每月總支出長條圖", text_auto='.2s', labels={'月份': '月份', '金額': '總金額 (NT$)'})
    fig_bar.update_traces(textangle=0, textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.subheader("消費類別分析")
    unique_months = sorted(df['月份'].unique(), reverse=True)
    if not unique_months:
        st.info("目前篩選的範圍內沒有可分析的月份。")
        return
    selected_month = st.selectbox("請選擇要分析的月份：", unique_months)
    if selected_month:
        month_df = df[df['月份'] == selected_month]
        category_summary = month_df.groupby('類別')['金額'].sum().reset_index()
        if not category_summary.empty:
            fig_pie = px.pie(category_summary, names='類別', values='金額', title=f"{selected_month} 月份消費佔比", hole=0.3)
            fig_pie.update_traces(textinfo='percent+label', pull=[0.05] * len(category_summary))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.subheader(f"{selected_month} 月份消費明細")
            details_df = month_df[['日期', '品項', '數量', '金額', '類別']].copy()
            details_df['日期'] = details_df['日期'].dt.strftime('%Y-%m-%d')
            details_df['數量'] = details_df['數量'].astype(int)
            details_df = details_df.sort_values(by='日期')
            st.dataframe(details_df, use_container_width=True, hide_index=True)
        else: st.write(f"{selected_month} 月份沒有可分析的資料。")

def page_edit_account(username):
    """編輯帳戶頁面"""
    st.title(f"✏️ 編輯 {username} 的帳戶")

    st.subheader("更換頭像")
    new_avatar_file = st.file_uploader("上傳新的頭像照片 (選填)", type=['png', 'jpg', 'jpeg'], key="avatar_uploader")
    if new_avatar_file:
        st.image(new_avatar_file, caption="新頭像預覽", width=150)

    st.subheader("變更密碼")
    new_password = st.text_input("設定您的新密碼 (留空則不變更)", type="password", key="new_pass")
    confirm_password = st.text_input("再次輸入您的新密碼", type="password", key="confirm_pass")

    st.write("---")

    if st.button("💾 儲存變更", use_container_width=True, type="primary"):
        if new_password and new_password != confirm_password:
            st.error("兩次輸入的新密碼不一致！")
        else:
            with st.spinner("正在更新您的帳戶資料..."):
                if new_password or new_avatar_file:
                    success, message = update_user(username, new_password if new_password else None, new_avatar_file)
                    if success:
                        st.success(message)
                        st.cache_data.clear()
                    else:
                        st.error(message)
                else:
                    st.info("沒有任何變更。")
    
    st.write("---")
    st.subheader("危險區域")
    
    # --- 程式碼變更處：使用 .lower() 進行不分大小寫的比對 ---
    if username.lower() == ADMIN_USERNAME.lower():
        st.warning("管理員模式：您可以刪除任何使用者帳號。此操作無法復原！")
        all_users = get_all_users()
        deletable_users = [user['username'] for user in all_users if user['username'].lower() != ADMIN_USERNAME.lower()]
        if deletable_users:
            user_to_delete = st.selectbox("選擇要刪除的使用者：", deletable_users)
            if st.button(f"刪除使用者「{user_to_delete}」", type="primary"):
                success, message = delete_user(user_to_delete)
                if success:
                    st.success(message)
                    st.cache_data.clear()
                    st.rerun() # 新增：刷新頁面
                else:
                    st.error(message)
        else:
            st.info("目前沒有其他可刪除的使用者。")
    else:
        delete_confirmation = st.checkbox(f"我了解這將永久刪除我的帳號「{username}」以及所有相關的消費紀錄。")
        if st.button("永久刪除我的帳號", disabled=not delete_confirmation, type="primary"):
            success, message = delete_user(username)
            if success:
                st.success(message)
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()
            else:
                st.error(message)

# --- 5. 主應用程式導覽與使用者登入 (Netflix 風格) ---

if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'selected_user' not in st.session_state: st.session_state.selected_user = None
if 'show_signup' not in st.session_state: st.session_state.show_signup = False

if not st.session_state.logged_in:
    st.title("誰正在使用 AI 發票記帳助理？")

    st.markdown("""
    <style>
        div[data-testid="stHorizontalBlock"] > div[data-testid^="stVerticalBlock"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.75rem;
        }
        div[data-testid="stImage"] img {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            object-fit: cover;
            border: 4px solid #444;
            transition: border-color 0.3s;
        }
        div[data-testid="stImage"] img:hover {
            border-color: #007bff;
        }
        div[data-testid="stButton"] {
            width: 150px;
        }
    </style>
    """, unsafe_allow_html=True)

    users = get_all_users()
    
    num_columns = 5
    cols = st.columns(num_columns)
    
    for i, user in enumerate(users):
        with cols[i % num_columns]:
            avatar_data = user.get('avatar_base64', '')
            if avatar_data:
                try:
                    img_data = base64.b64decode(avatar_data)
                    st.image(img_data, width=150)
                except:
                    st.image("https://placehold.co/150x150/4A4A4A/FFFFFF?text=👤", width=150)
            else:
                st.image("https://placehold.co/150x150/4A4A4A/FFFFFF?text=👤", width=150)
            
            if st.button(str(user.get('username', '')), key=f"user_{user.get('username', i)}", use_container_width=True):
                st.session_state.selected_user = user.get('username')
                st.session_state.show_signup = False
                st.rerun()

    with cols[len(users) % num_columns]:
        st.image("https://placehold.co/150x150/FFFFFF/000000?text=➕", width=150)
        if st.button("新增使用者", key="add_user", use_container_width=True):
            st.session_state.show_signup = True
            st.session_state.selected_user = None
            st.rerun()

    if st.session_state.selected_user and not st.session_state.show_signup:
        st.write("---")
        with st.form("password_form"):
            st.subheader(f"你好, {st.session_state.selected_user}！請輸入密碼登入。")
            password = st.text_input("密碼", type="password", key="password_input")
            submitted = st.form_submit_button("登入", use_container_width=True, type="primary")

            if submitted:
                # --- 程式碼變更處：使用新的 check_login 回傳值 ---
                login_success, correct_username = check_login(st.session_state.selected_user, password)
                if login_success:
                    st.session_state.logged_in = True
                    st.session_state.username = correct_username # 儲存正確大小寫的名稱
                    st.session_state.selected_user = None
                    st.rerun()
                else:
                    st.error("密碼錯誤！")

    if st.session_state.show_signup:
        st.write("---")
        with st.form("signup_form"):
            st.subheader("建立新的使用者")
            new_username = st.text_input("設定您的使用者名稱")
            new_password = st.text_input("設定您的密碼", type="password")
            confirm_password = st.text_input("再次輸入您的密碼", type="password")
            avatar_file = st.file_uploader("上傳您的頭像照片 (選填)", type=['png', 'jpg', 'jpeg'])
            
            signup_submitted = st.form_submit_button("註冊", use_container_width=True)

            if signup_submitted:
                if not new_username or not new_password:
                    st.warning("使用者名稱和密碼不能為空！")
                elif new_password != confirm_password:
                    st.warning("兩次輸入的密碼不一致！")
                else:
                    success, message = add_user(new_username, new_password, avatar_file)
                    if success:
                        st.success(message)
                        st.session_state.show_signup = False
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(message)

elif st.session_state.logged_in:
    username = st.session_state.username
    
    with st.sidebar:
        st.header(f"你好, {username}！")
        st.write("---")
        st.header("功能選單")
        
        if 'page' not in st.session_state:
            st.session_state.page = '發票辨識'

        if st.button("發票辨識", use_container_width=True, type="secondary" if st.session_state.page != '發票辨識' else "primary"):
            st.session_state.page = '發票辨識'
        if st.button("消費儀表板", use_container_width=True, type="secondary" if st.session_state.page != '儀表板' else "primary"):
            st.session_state.page = '儀表板'
        if st.button("編輯帳戶", use_container_width=True, type="secondary" if st.session_state.page != '編輯帳戶' else "primary"):
            st.session_state.page = '編輯帳戶'
        
        st.write("---")
        if st.button("登出", use_container_width=True):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    if st.session_state.page == '發票辨識':
        page_invoice_processing(username)
    elif st.session_state.page == '儀表板':
        page_dashboard(username)
    elif st.session_state.page == '編輯帳戶':
        page_edit_account(username)
