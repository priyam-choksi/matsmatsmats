import requests
API_KEY = "d3g66kpr01qqbh55hfb0d3g66kpr01qqbh55hfbg"
API_ENDPOINT = "https://finnhub.io/api/v1/quote"
# Test the API key with a sample request
response = requests.get(f"{API_ENDPOINT}?symbol=AAPL&token={API_KEY}")
if response.status_code == 200:
   print("API Key is working!")
   print(response.json()) # Display the response data
else:
   print("API Key is invalid or expired.")
   print(f"Error: {response.status_code}, {response.text}")