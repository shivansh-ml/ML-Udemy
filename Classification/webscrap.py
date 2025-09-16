# import requests
# response = requests.get('https://www.geeksforgeeks.org/python/python-programming-language-tutorial/')
# # print(response.status_code)
# # print(response.content)
# from bs4 import BeautifulSoup
# soup = BeautifulSoup(response.content, 'html.parser')
# print(soup.prettify())

# import requests
# from bs4 import BeautifulSoup

# # List of URLs to scrape
# urls_to_scrape = [
#     'https://www.geeksforgeeks.org/python/python-programming-language-tutorial/',
#     'https://scikit-learn.org/stable/modules/ensemble.html#',
#     'https://www.theguardian.com/us-news/2025/sep/15/elon-musk-tesla-stock'
#     'https://www.octoparse.com/blog/top-10-most-scraped-websites#top-8-indeed'
#     'https://dev.to/mohamednizzad/dreamnestai-ai-powered-house-design-2d-3d-plan-audio-video-walkthroughs-smart-e-commerce-16i6'
#     'https://dev.to/aws/building-production-ready-ai-agents-with-llamaindex-and-amazon-bedrock-agentcore-1fm3'
#     'https://towardsdatascience.com/no-peeking-ahead-time-aware-graph-fraud-detection/'
#     'https://dev.to/alifar/gpt-5-codex-why-openais-new-model-matters-for-developers-2e5g'
#     'https://towardsdatascience.com/docling-the-document-alchemist/'
#     'https://dev.to/mrzaizai2k/til-building-a-simple-qr-generator-5e37'
#     # Add 17 more URLs here
# ]

# # Open the file in append mode ('a')
# with open('scraped_content.txt', 'a', encoding='utf-8') as file:
#     for url in urls_to_scrape:
#         print(f"Scraping content from: {url}")
        
#         try:
#             # Fetch the page
#             response = requests.get(url, timeout=10) # Added a timeout for safety
#             response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            
#             # Parse the page
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             # Find the main content container (this part may need to be adjusted for each website)
#             # A more robust solution might involve a function to handle different site structures
#             content_div = soup.find('div', class_='article--viewer_content') 

#             # Write a separator to distinguish content from different URLs
#             file.write(f"\n\n--- Content from {url} ---\n\n")
            
#             if content_div:
#                 for para in content_div.find_all('p'):
#                     file.write(para.text.strip() + '\n')
#             else:
#                 file.write("No article content found or selector not valid for this URL.\n")
                
#         except requests.exceptions.RequestException as e:
#             print(f"Could not scrape {url}: {e}")
#             file.write(f"--- Failed to scrape {url} due to an error ---\n")
            
# print("\nScraping complete. All content has been appended to scraped_content.txt")

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# List of URLs to scrape
urls_to_scrape = [
    'https://www.hindustantimes.com/india-news/nano-banana-trend-10-tips-and-tricks-to-generate-the-best-pics-using-gemini-ai-101757899785637.html'     # Another good example
    
    # Add other URLs from your list
]

# Set up the Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # This is the key part - runs the browser without a UI
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open the file in append mode ('a')
with open('scraped_content.txt', 'a', encoding='utf-8') as file:
    for url in urls_to_scrape:
        print(f"Scraping content from: {url}")
        
        try:
            driver.get(url)
            
            # You might need to wait for the JavaScript to load. 
            # A simple sleep is often enough, but explicit waits are better.
            time.sleep(5) 
            
            # Get the fully rendered page source
            page_source = driver.page_source
            
            # Pass the page source to BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Write a separator to distinguish content from different URLs
            file.write(f"\n\n--- Content from {url} ---\n\n")
            
            # Now, find and scrape the content as before. 
            # Note: The CSS selectors might be different now.
            # Example: Find all paragraph tags
            content_paragraphs = soup.find_all('p')
            
            if content_paragraphs:
                for para in content_paragraphs:
                    file.write(para.text.strip() + '\n')
            else:
                file.write("No content found.\n")
                
        except Exception as e:
            print(f"Could not scrape {url}: {e}")
            file.write(f"--- Failed to scrape {url} due to an error ---\n")
            
# Don't forget to close the browser!
driver.quit()
print("\nScraping complete. All content has been appended to scraped_content.txt")