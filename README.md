## 🚀 How to Run the Project

Follow these steps to install dependencies and run the Cryptocurrency Analytics Dashboard locally.

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

Navigate to the project directory:

```bash
cd CryptoAnalytics
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Required Libraries

Install the required Python packages:

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly
```

Alternatively, create a `requirements.txt` file with the following dependencies:

```text
streamlit
pandas
numpy
scikit-learn
xgboost
plotly
```

Install them using:

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Place the cryptocurrency dataset named `all_currencies.csv` in the project root directory.

The project folder should look like this:

```text
CryptoAnalytics/
│
├── app.py
├── all_currencies.csv
├── requirements.txt
└── README.md
```

### 5. Run the Application

Execute the following command in your terminal:

```bash
streamlit run app.py
```

Alternatively:

```bash
python -m streamlit run app.py
```

### 6. Access the Dashboard

After running the application, open your web browser and visit:

```text
http://localhost:8501
```

The Cryptocurrency Analytics Dashboard will open in your browser.

### 7. Using the Application

1. Upload `all_currencies.csv` through the sidebar, or place it in the project directory.
2. Select a cryptocurrency from the dropdown menu (BTC, ETH, etc.).
3. Click **Run Analysis & Train Models**.
4. Explore the Market Overview dashboard.
5. View technical indicators such as RSI, MACD and moving averages.
6. Compare Random Forest and XGBoost model performance.
7. View next-day price predictions and BUY/SELL/HOLD signals.
8. Download processed datasets and prediction results.

### 8. Stop the Application

To stop the Streamlit application, press:

```bash
Ctrl + C
```

### ⚠️ Important Notes

- Python must
