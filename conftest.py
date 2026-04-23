import sys
import os
import asyncio
from dotenv import load_dotenv

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=env_path, override=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

key = os.getenv("GOOGLE_API_KEY")
print(f"\n✅ GOOGLE_API_KEY loaded: {'YES - ' + key[:10] + '...' if key else 'NO - KEY IS MISSING!'}\n")