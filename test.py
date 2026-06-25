"""Small test script to verify yfinance download with a browser-like session.

This file demonstrates constructing a `requests.Session` with common browser
headers and passing it to `yfinance.download` to reduce request blocking.
"""

import yfinance as yf
from requests import Session

# Create a custom network session with browser headers to reduce chance of blocking
session = Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
})

# Pass the browser-like session into yfinance and download recent AAPL data
df = yf.download('AAPL', start='2025-01-01', session=session)