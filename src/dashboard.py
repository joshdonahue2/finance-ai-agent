"""
Streamlit Dashboard: PDF Upload, Insights, and Visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
from dotenv import load_dotenv

import requests
import json

# Compatible imports for supabase 1.2.0 and ollama 0.1.7
from supabase import create_client, Client
import ollama

# Import custom modules
try:
    from process_pdf import PDFProcessor
    from rag_query import get_relevant_chunks, generate_insights
except ImportError as e:
    print(f"Import error: {e}")  # Use print instead of st.error for startup
    st.error(f"Failed to import modules: {e}")
    raise e
    # Create dummy objects to prevent further errors
    class DummyProcessor:
        def process_single_pdf(self, *args, **kwargs): return 0
    processor = DummyProcessor()

# Load environment
load_dotenv()

# Page config
st.set_page_config(
    page_title="💰 Finance AI Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)
def get_ollama_models():
    """Fetch the list of models from the Ollama APIs."""
    models = []
    
    ocr_host = os.getenv("OLLAMA_HOST_OCR")
    chat_host = os.getenv("OLLAMA_HOST_CHAT")

    if ocr_host:
        try:
            response = requests.get(f"{ocr_host}/api/tags")
            response.raise_for_status()
            ocr_models = response.json()["models"]
            models.extend([f"ocr::{model['name']}" for model in ocr_models])
        except (requests.exceptions.RequestException, KeyError) as e:
            st.error(f"Could not fetch models from OCR host: {e}")

    if chat_host:
        try:
            response = requests.get(f"{chat_host}/api/tags")
            response.raise_for_status()
            chat_models = response.json()["models"]
            models.extend([f"chat::{model['name']}" for model in chat_models])
        except (requests.exceptions.RequestException, KeyError) as e:
            st.error(f"Could not fetch models from Chat host: {e}")
            
    # Add default single host if the specific ones are not defined
    if not ocr_host and not chat_host:
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            response = requests.get(f"{ollama_host}/api/tags")
            response.raise_for_status()
            default_models = response.json()["models"]
            models.extend([model["name"] for model in default_models])
        except (requests.exceptions.RequestException, KeyError) as e:
            st.error(f"Could not fetch Ollama models: {e}")

    return models

# Initialize Supabase client (FIXED: Use only os.getenv, no st.secrets)
def init_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        st.error("❌ **Missing Supabase Configuration**\n\nPlease check your `.env` file contains `SUPABASE_URL` and `SUPABASE_KEY`.")
        st.stop()
        return None
    
    try:
        client = create_client(url, key)
        # Test connection
        response = client.table('transactions').select('count', count='exact').execute()
        st.success(f"✅ **Supabase Connected!** Found {response.count} transactions")
        return client
    except Exception as e:
        st.error(f"❌ **Supabase Connection Failed**: {str(e)[:200]}...\n\n**Check:**\n• SUPABASE_URL is correct\n• SUPABASE_KEY is valid\n• Network access to your Supabase instance")
        st.stop()
        return None

# Initialize global client
client = init_client()
if client:
    processor = PDFProcessor(client)
else:
    processor = None
    st.error("Supabase client could not be initialized. Please check your credentials in the `.env` file and restart the application.")

# Custom CSS for beauty
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric > label {
        color: white !important;
        font-size: 1.2rem;
    }
    .stMetric > div > div {
        color: white !important;
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def fetch_transactions() -> pd.DataFrame:
    """Fetch all transactions from Supabase"""
    try:
        response = client.table('transactions').select('*').execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            df['amount'] = pd.to_numeric(df['amount'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()

def categorize_transactions(df: pd.DataFrame, chat_model: str) -> pd.DataFrame:
    """Auto-categorize transactions using LLM"""
    if df.empty:
        return df
    
    # Determine host and model name
    if "::" in chat_model:
        instance, model_name = chat_model.split("::")
        if instance == "chat":
            host = os.getenv("OLLAMA_HOST_CHAT")
        else: # Should not happen based on UI filtering
            host = os.getenv("OLLAMA_HOST_OCR")
    else: # No prefix, use default host
        model_name = chat_model
        host = os.getenv("OLLAMA_HOST")

    ollama_client = ollama.Client(host=host)

    progress_bar = st.progress(0)
    categories = []
    
    for idx, row in df.iterrows():
        if pd.isna(row.get('category')) or not row['category']:
            prompt = f"Categorize this transaction: '{row['description']}' (amount: ${row['amount']:.2f})\nChoose one: Groceries, Dining, Transportation, Utilities, Entertainment, Shopping, Bills, Income, Transfer, Other"
            
            try:
                response = ollama_client.generate(model=model_name, prompt=prompt)
                category = response['response'].strip().split()[0]  # Take first word
                categories.append(category)
            except:
                categories.append("Other")
        else:
            categories.append(row['category'])
        
        progress_bar.progress((idx + 1) / len(df))
    
    df['category'] = categories
    
    # Update database
    # Use the global supabase client, not the ollama client
    global client
    for _, row in df.iterrows():
        if pd.notna(row['id']):
            client.table('transactions').update({
                'category': row['category'],
                'updated_at': datetime.now().isoformat()
            }).eq('id', row['id']).execute()
    
    return df

# Header
st.markdown('<h1 class="main-header">💰 Personal Finance AI Agent</h1>', unsafe_allow_html=True)

# Sidebar: Upload and Controls
with st.sidebar:
    st.header("📁 Document Upload")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF bank statement",
        type="pdf",
        help="Upload scanned bank statements from USAA or Chase"
    )
    
    if uploaded_file is not None:
        # Process button
        if st.button("🚀 Process & Analyze", type="primary"):
            with st.spinner("Processing your statement... This may take 1-2 minutes."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Process the PDF
                    count = processor.process_single_pdf(tmp_path, ocr_model=st.session_state.ocr_model, chat_model=st.session_state.chat_model)
                    st.success(f"✅ Successfully processed {count} transactions!")
                    
                    # Refresh data
                    st.cache_data.clear()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                
                finally:
                    # Clean up
                    os.unlink(tmp_path)
    
    st.header("⚙️ Settings")

    # Model Selection
    available_models = get_ollama_models()
    
    multimodal_models = [m for m in available_models if m.startswith("ocr::") or ("llava" in m and "::" not in m) or ("gemma" in m and "::" not in m) or ("mistral" in m and "::" not in m)]
    multimodal_models.append("Mistral API")
    chat_models = [m for m in available_models if m.startswith("chat::") or ("::" not in m)]

    st.session_state.ocr_model = st.selectbox(
        "Select OCR Model",
        multimodal_models,
        index=0 if multimodal_models else -1,
        help="Select a multimodal model for PDF processing."
    )

    st.session_state.chat_model = st.selectbox(
        "Select Chat Model",
        chat_models,
        index=chat_models.index("chat::llama3") if "chat::llama3" in chat_models else (chat_models.index("llama3") if "llama3" in chat_models else 0),
        help="Select a model for chat and insights."
    )

    refresh_data = st.button("🔄 Refresh Data")
    auto_categorize = st.checkbox("Auto-categorize on load", value=True)

# Main content
col1, col2, col3 = st.columns(3)
total_spent = 0
transaction_count = 0
avg_monthly = 0

# Fetch and process data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_and_process_data():
    df = fetch_transactions()
    
    if df.empty:
        # Return empty defaults when no data
        return pd.DataFrame(), 0.0, 0, 0.0
    
    # Basic metrics
    total_spent = abs(df['amount'].sum())
    transaction_count = len(df)
    months = (df['transaction_date'].max() - df['transaction_date'].min()).days / 30
    avg_monthly = total_spent / max(months, 1)
    
    # Auto-categorize if requested
    if 'auto_categorize' in globals() and auto_categorize:  # Check if defined
        df = categorize_transactions(df, chat_model=st.session_state.chat_model)
    
    return df, total_spent, transaction_count, avg_monthly

# Initialize with safe unpacking
try:
    df, total_spent, transaction_count, avg_monthly = load_and_process_data()
except ValueError:
    # Fallback for empty data
    df = pd.DataFrame()
    total_spent = 0.0
    transaction_count = 0
    avg_monthly = 0.0

if refresh_data:
    st.cache_data.clear()
    st.rerun()

# Metrics
if not df.empty:
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Spent", f"${total_spent:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Transactions", f"{transaction_count:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Monthly", f"${avg_monthly:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

# Main dashboard
if df.empty:
    st.info("👋 Welcome! Upload your first bank statement using the sidebar to get started.")
    st.balloons()
else:
    # Data quality check
    recent_data = df['transaction_date'].max()
    if datetime.now() - recent_data > timedelta(days=90):
        st.warning("📊 Your most recent data is over 90 days old. Consider uploading newer statements.")
    
    from budget import budget_page
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Overview", "💰 Budget", "📝 Transactions", "🔍 Insights", "📋 Anomalies", "⚙️ Batch Processing"])
    
    with tab1:
        # Overview visualizations
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Spending by category pie chart
            if 'category' in df.columns and df['category'].notna().sum() > 0:
                cat_spending = df[df['category'].notna()].groupby('category')['amount'].sum().abs()
                fig_pie = px.pie(
                    values=cat_spending.values,
                    names=cat_spending.index,
                    title="Spending by Category",
                    color_discrete_sequence=px.colors.sequential.Sunset
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("ℹ️ Categorize transactions to see this chart")
        
        with col_right:
            # Monthly trends
            df['month'] = df['transaction_date'].dt.to_period('M')
            monthly_spending = df.groupby('month')['amount'].sum().abs().reset_index()
            monthly_spending['month'] = monthly_spending['month'].astype(str)
            
            fig_line = px.line(
                monthly_spending, 
                x='month', 
                y='amount',
                title="Monthly Spending Trends",
                markers=True,
                color_discrete_sequence=['#636EFA']
            )
            fig_line.update_layout(xaxis_title="Month", yaxis_title="Spending ($)")
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Bank comparison
        if len(df['bank_name'].unique()) > 1:
            bank_spending = df.groupby('bank_name')['amount'].sum().abs().reset_index()
            fig_bank = px.bar(
                bank_spending,
                x='bank_name',
                y='amount',
                title="Spending by Bank",
                color='amount',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_bank, use_container_width=True)

    with tab2:
        budget_page()

    with tab3:
        st.header("📝 Edit Transactions")
        
        # Define which columns should be editable
        editable_cols = {
            "category": st.column_config.SelectboxColumn(
                "Category",
                options=[
                    "Groceries", "Dining", "Transportation", "Utilities", 
                    "Entertainment", "Shopping", "Bills", "Income", 
                    "Transfer", "Other"
                ],
                required=True,
            ),
            "description": st.column_config.TextColumn("Description"),
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
        }
        
        # Keep a copy of the original df to compare against
        if 'original_df' not in st.session_state:
            st.session_state.original_df = df.copy()

        edited_df = st.data_editor(
            df,
            column_config=editable_cols,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="data_editor"
        )

        if st.button("💾 Save Changes"):
            with st.spinner("Saving changes..."):
                # Find changes by comparing the edited df with the original
                changes = pd.concat([st.session_state.original_df, edited_df]).drop_duplicates(keep=False)
                
                if not changes.empty:
                    for _, row in changes.iterrows():
                        try:
                            update_data = {
                                'category': row['category'],
                                'description': row['description'],
                                'amount': row['amount'],
                                'updated_at': datetime.now().isoformat()
                            }
                            # Update the corresponding row in Supabase
                            client.table('transactions').update(update_data).eq('id', row['id']).execute()
                        except Exception as e:
                            st.error(f"Failed to update transaction {row['id']}: {e}")
                    
                    st.success(f"✅ Successfully saved {len(changes)} changes!")
                    # Clear cache and rerun to show updated data
                    st.cache_data.clear()
                    st.session_state.original_df = edited_df.copy() # Update original_df
                    st.rerun()
                else:
                    st.info("No changes to save.")
    
    with tab4:
        st.header("🤖 AI-Powered Insights")
        
        # Query interface
        query = st.text_input(
            "Ask me anything about your finances...",
            placeholder="e.g., 'Show me my biggest expenses last quarter' or 'What should I budget for groceries?'",
            help="The AI will analyze your transaction data to provide personalized insights"
        )
        
        if query:
            with st.spinner("Analyzing your finances..."):
                # RAG: Retrieve relevant transactions
                context = get_relevant_chunks(query, client, chat_model=st.session_state.chat_model)
                
                if context:
                    # Generate insights
                    insights = generate_insights(query, context, model=st.session_state.chat_model)
                    
                    # Display with markdown
                    st.markdown("### 💡 AI Insights")
                    st.markdown(insights)
                    
                    # Show source transactions
                    with st.expander("📄 Source Transactions"):
                        st.text(context[:2000] + "..." if len(context) > 2000 else context)
                else:
                    st.warning("No relevant transactions found for this query.")
        
        # Quick insights buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Spending Summary"):
                summary_prompt = "Provide a high-level summary of my spending patterns, biggest categories, and overall financial health."
                context = get_relevant_chunks(summary_prompt, client, chat_model=st.session_state.chat_model)
                if context:
                    insights = generate_insights(summary_prompt, context, model=st.session_state.chat_model)
                    st.markdown(insights)
        
        with col2:
            if st.button("🎯 Budget Recommendations"):
                budget_prompt = "Based on my transaction history, suggest a realistic monthly budget by category and areas to cut back."
                context = get_relevant_chunks(budget_prompt, client, chat_model=st.session_state.chat_model)
                if context:
                    insights = generate_insights(budget_prompt, context, model=st.session_state.chat_model)
                    st.markdown(insights)
        
        with col3:
            if st.button("💼 Tax Deductions"):
                tax_prompt = "Identify potential tax-deductible expenses from my transactions (home office, business expenses, charitable donations, etc.)."
                context = get_relevant_chunks(tax_prompt, client, chat_model=st.session_state.chat_model)
                if context:
                    insights = generate_insights(tax_prompt, context, model=st.session_state.chat_model)
                    st.markdown(insights)
    
    with tab5:
        st.header("🚨 Anomaly Detection")
        
        # Statistical anomalies - FIXED
        avg_amount = df['amount'].abs().mean()
        std_amount = df['amount'].abs().std()
        
        # Flag transactions > 2 standard deviations (use absolute values)
        df['is_anomaly'] = df['amount'].abs() > (avg_amount + 2 * std_amount)
        anomalies = df[df['is_anomaly']].copy()
        
        if not anomalies.empty:
            st.success(f"🔍 Found {len(anomalies)} potential anomalies!")
            
            # Anomaly table
            anomaly_cols = ['transaction_date', 'description', 'amount', 'category', 'bank_name']
            st.dataframe(
                anomalies[anomaly_cols].sort_values('amount', key=abs, ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            # Anomaly visualization
            anomalies['size_abs'] = anomalies['amount'].abs()
            fig_anomaly = px.scatter(
                anomalies, 
                x='transaction_date',
                y='amount',
                size='size_abs',  # Use absolute value for bubble size
                color='category',
                hover_data=['description', 'bank_name'],
                title="Anomalous Transactions",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_anomaly.add_hline(y=avg_amount + 2 * std_amount, line_dash="dash", 
                                  line_color="red", annotation_text="Anomaly Threshold")
            st.plotly_chart(fig_anomaly, use_container_width=True)
        else:
            st.info("✅ No statistical anomalies detected! Your spending looks consistent.")
        
        # Manual anomaly review
        st.subheader("🔍 Review Large Transactions")
        large_threshold = st.slider("Show transactions over $", 100, 1000, 250)
        large_transactions = df[df['amount'].abs() > large_threshold].sort_values('amount', key=abs, ascending=False)
        
        if not large_transactions.empty:
            st.dataframe(large_transactions[['transaction_date', 'description', 'amount', 'category']], 
                        use_container_width=True, hide_index=True)
    
    with tab6:
        st.header("📦 Batch Processing")
        st.info("Use this tab to process multiple PDFs at once (e.g., your initial 7 years of data)")
        
        # Initial batch upload
        uploaded_files = st.file_uploader(
            "Upload multiple PDFs",
            type="pdf",
            accept_multiple_files=True,
            help="Upload all your historical statements at once"
        )
        
        if uploaded_files:
            if st.button(f"🚀 Process {len(uploaded_files)} files", type="primary"):
                with st.spinner(f"Processing {len(uploaded_files)} files..."):
                    total_count = 0
                    for i, uploaded_file in enumerate(uploaded_files):
                        st.info(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            bank_name = processor.detect_bank_from_filename(uploaded_file.name)
                            count = processor.process_single_pdf(tmp_path)
                            total_count += count
                        except Exception as e:
                            st.error(f"Failed to process {uploaded_file.name}: {e}")
                        finally:
                            os.unlink(tmp_path)
                    
                    st.success(f"✅ Batch complete! Processed {total_count} total transactions.")
                    st.cache_data.clear()
                    st.rerun()
        
        # Folder monitoring (advanced)
        st.subheader("📁 Folder Processing")
        folder_path = st.text_input("Local folder path", value="./initial_pdfs")
        
        if st.button("Process folder contents"):
            try:
                # This would run the batch processor
                result = subprocess.run(
                    ["python", "src/process_pdf.py", "--batch", folder_path],
                    capture_output=True, text=True, cwd=os.getcwd()
                )
                st.code(result.stdout)
                if result.stderr:
                    st.error(result.stderr)
                st.success("Batch folder processing complete!")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Batch processing failed: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit, Ollama, and Supabase | All data stays local</p>",
    unsafe_allow_html=True
)