import os
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

logger.debug(f"Initializing Supabase with URL: {supabase_url}")

supabase: Client = create_client(supabase_url, supabase_key)

def init_db():
    """
    Initialize database tables if they don't exist
    Note: In Supabase, you need to create tables through the dashboard or SQL editor
    """
    pass  # Tables are created in Supabase dashboard

def get_db():
    """
    Get Supabase client instance
    """
    return supabase 