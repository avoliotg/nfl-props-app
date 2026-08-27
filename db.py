"""Database connection + helpers for OpalScales (Supabase)."""
import streamlit as st
from supabase import create_client


@st.cache_resource
def get_client():
    """Create a cached Supabase client from secrets."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def test_connection():
    """Quick check: can we reach the database? Returns (ok, message)."""
    try:
        url = st.secrets["SUPABASE_URL"]
        client = get_client()
        resp = client.table("lines").select("*").limit(1).execute()
        return True, f"Connected! ({len(resp.data)} rows)"
    except Exception as e:
        return False, f"Connection failed: {e} | URL used: {st.secrets['SUPABASE_URL']}"