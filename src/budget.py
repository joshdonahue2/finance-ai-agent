import streamlit as st
import pandas as pd
from supabase import create_client, Client
import os
from datetime import datetime

# Initialize Supabase client
@st.cache_resource
def init_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

client = init_client()

def fetch_budgets():
    """Fetch all budgets from Supabase"""
    try:
        response = client.table('budgets').select('*').execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame(columns=['category', 'budgeted_amount'])
    except Exception as e:
        st.error(f"Failed to fetch budgets: {e}")
        return pd.DataFrame(columns=['category', 'budgeted_amount'])

def fetch_transactions_for_month(month):
    """Fetch all transactions for a given month from Supabase"""
    try:
        # Assuming 'month' is a string like '2023-10'
        start_date = f"{month}-01"
        end_date = pd.to_datetime(start_date).to_period('M').end_time.strftime('%Y-%m-%d')
        
        response = client.table('transactions').select('*').gte('transaction_date', start_date).lte('transaction_date', end_date).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df['amount'] = pd.to_numeric(df['amount'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to fetch transactions: {e}")
        return pd.DataFrame()

def budget_page():
    st.header("💰 Monthly Budget")

    # Month selector
    current_month = datetime.now().strftime("%Y-%m")
    month_to_show = st.selectbox("Select Month", [current_month, (datetime.now() - pd.DateOffset(months=1)).strftime("%Y-%m")])

    # Fetch data
    budgets_df = fetch_budgets()
    transactions_df = fetch_transactions_for_month(month_to_show)

    # Spending by category
    if not transactions_df.empty:
        spending_by_category = transactions_df.groupby('category')['amount'].sum().abs().reset_index()
        spending_by_category.rename(columns={'amount': 'spent_amount'}, inplace=True)
    else:
        spending_by_category = pd.DataFrame(columns=['category', 'spent_amount'])

    # Merge budgets and spending
    if not budgets_df.empty:
        budget_overview = pd.merge(budgets_df, spending_by_category, on='category', how='left').fillna(0)
        budget_overview['remaining'] = budget_overview['budgeted_amount'] - budget_overview['spent_amount']
        budget_overview['progress'] = (budget_overview['spent_amount'] / budget_overview['budgeted_amount']).clip(0, 1)
    else:
        budget_overview = pd.DataFrame(columns=['category', 'budgeted_amount', 'spent_amount', 'remaining', 'progress'])


    st.subheader("Budget Overview")
    if not budget_overview.empty:
        for _, row in budget_overview.iterrows():
            st.markdown(f"**{row['category']}**")
            st.progress(row['progress'])
            st.markdown(f"Spent: ${row['spent_amount']:,.2f} of ${row['budgeted_amount']:,.2f} | Remaining: ${row['remaining']:,.2f}")
    else:
        st.info("No budgets set. Go to 'Set Budgets' to create your first budget.")


    st.subheader("Set Budgets")
    
    categories = ["Groceries", "Dining", "Transportation", "Utilities", "Entertainment", "Shopping", "Bills", "Income", "Transfer", "Other"]
    
    with st.form("budget_form"):
        for category in categories:
            default_budget = budgets_df[budgets_df['category'] == category]['budgeted_amount'].iloc[0] if not budgets_df[budgets_df['category'] == category].empty else 0.0
            st.number_input(f"Budget for {category}", key=f"budget_{category}", value=default_budget, min_value=0.0, step=50.0)
        
        submitted = st.form_submit_button("Save Budgets")
        if submitted:
            with st.spinner("Saving budgets..."):
                for category in categories:
                    budgeted_amount = st.session_state[f"budget_{category}"]
                    
                    # Check if budget for category already exists
                    existing_budget = budgets_df[budgets_df['category'] == category]
                    
                    if not existing_budget.empty:
                        # Update existing budget
                        client.table('budgets').update({'budgeted_amount': budgeted_amount, 'updated_at': datetime.now().isoformat()}).eq('category', category).execute()
                    else:
                        # Insert new budget
                        client.table('budgets').insert({'category': category, 'budgeted_amount': budgeted_amount}).execute()

            st.success("Budgets saved successfully!")
            st.rerun()

if __name__ == '__main__':
    budget_page()