# Data Collection Plan for TransferIQ Project

### 1. Player Performance Data
- **Purpose:** Obtain detailed match-level statistics for football players.  
- **Main Source:** *StatsBomb Open Data* – a free, professional-grade dataset covering various competitions.  
- **Approach:**  
  1. **Use the `statsbombpy` Python library** – the most convenient way to directly access StatsBomb data.  
  2. **Setup:** Install with `pip install statsbombpy`.  
  3. **Steps for Extraction:**  
     - Import the library into your Python script.  
     - List all available competitions to decide which ones to analyze.  
     - Select a competition and fetch the corresponding match list.  
     - For each match, download the event-level data (e.g., passes, shots, tackles, dribbles, including pitch coordinates).  
     - Save the collected data (commonly as pandas DataFrames) to enable downstream cleaning and feature engineering.  

---

### 2. Market Value Data
- **Purpose:** Gather both historical and current player market valuations.  
- **Main Source:** *Transfermarkt.com*.  
- **Approach:** Web scraping using Python.  
  1. **Required Libraries:** `requests` for HTTP calls and `BeautifulSoup4` for parsing HTML. Install via: `pip install requests beautifulsoup4`.  
  2. **Scraping Workflow:**  
     - Build a list of URLs pointing to player profile pages on Transfermarkt.  
     - Fetch page content using `requests.get(url)`.  
     - Create a `BeautifulSoup` object from the downloaded HTML.  
     - With browser dev tools (Inspect Element), locate HTML tags and classes containing the market value info.  
     - Extract the values from the parsed object.  
     - Loop over all player URLs to collect values systematically.  
  3. **Ethical Practices:** Review `robots.txt` before scraping, and include pauses (`time.sleep()`) between requests to avoid stressing the server.  

---

### 3. Social Media Sentiment Data
- **Purpose:** Measure public opinion and sentiment about players from social media discussions.  
- **Main Source:** *Twitter (X)* through its official API.  
- **Approach:** Using Python with the Twitter API.  
  1. **Setup Requirements:** Obtain a Twitter Developer Account and generate API keys and tokens for authentication.  
  2. **Library to Use:** `Tweepy`, which simplifies API calls. Install with `pip install tweepy`.  
  3. **Data Collection Steps:**  
     - Authenticate the script with API credentials.  
     - Query tweets mentioning the player’s name, handle, or hashtags (e.g., `#Messi`).  
     - Collect tweet text along with metadata (date, likes, retweets).  
     - Store the raw text for later NLP-based sentiment scoring (e.g., using VADER or TextBlob).  

---

### 4. Injury Data
- **Purpose:** Build a dataset of player injury histories, including type, severity, and duration.  
- **Main Sources:** Not centralized; can use Transfermarkt’s *injury tab* on player profiles, sports portals, or existing datasets.  
- **Approach:** Combination of scraping and dataset search.  
  1. **Web Scraping:** Similar to the Transfermarkt market value workflow. Identify a reliable site that tracks injuries and extract details such as injury type, start/end dates, expected return, and matches missed.  
  2. **Searching for Datasets:** Before implementing scrapers, check platforms like Kaggle for precompiled injury datasets, which could save effort.  
  3. **Challenges:** Injury data is often unstructured and inconsistent, making preprocessing and cleaning especially critical.  
