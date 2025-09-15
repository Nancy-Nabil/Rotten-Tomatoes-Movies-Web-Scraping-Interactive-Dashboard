Rotten Tomatoes Movies — Web Scraping & Interactive Dashboard
Project Overview
This project collects, cleans, and analyzes movie data from Rotten Tomatoes, then presents it in an interactive Streamlit dashboard. It is designed to demonstrate a full data pipeline: scraping → cleaning → exploratory analysis → visualization → deployment.

Features
Web Scraping: Extracts movie details (titles, genres, year, critic & audience scores).
Data Cleaning: Removes duplicates, handles missing values, normalizes formats.
Exploratory Analysis: Includes visualizations of trends by year, genre distributions, and critic vs. audience comparisons.
Interactive Dashboard: Built with Streamlit; allows filtering, exploring, and downloading data.
Deployment Ready: Can be run locally or deployed via Streamlit Cloud.
Repository Structure
README.md
requirements.txt
data/
  raw/            # Raw scraped CSVs
  processed/      # Cleaned datasets
notebooks/
  scraping_and_analysis.ipynb
app/
  app.py          # Streamlit dashboard
  utils.py        # Data cleaning helpers
scripts/
  scrape_rt.py    # Rotten Tomatoes scraper
tests/
  test_cleaning.py

Clean the dataset
python app/utils.py
This will output data/processed/movies_clean.csv.

Run the dashboard
streamlit run app/app.py
Insights (Sample)
Critics tend to give lower ratings than audiences for blockbuster action movies.
Independent dramas often score higher with critics than general audiences.
Movie release trends show a surge in thrillers and horror post-2015.
Deployment
The dashboard can be deployed easily to Streamlit Cloud:

🙌 Acknowledgements
Data from Rotten Tomatoes
Built with Python, BeautifulSoup, Pandas, Plotly, and Streamlit
👩‍💻 Author
Nancy Nabil Mohamed Emam

Email: nancnanbil647@gmail.com

LinkedIn: Nancy Nabil
