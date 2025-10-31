# Portfolio Optimization Dashboard

A Python-based dashboard for **portfolio optimization** using historical asset prices to calculate **efficient frontiers**, **portfolio risk and return**, and **Sharpe ratios**. Built with **Streamlit**, **pandas**, and **NumPy**.

💻 **Live Demo:** [Streamlit App](https://portfolio-optimization-dashboard-mahyar-mirzaei.streamlit.app/)

---

## Features

- Visualize historical asset prices
- Calculate expected returns and volatility
- Plot the **efficient frontier**
- Compute portfolio weights for optimal risk-return combinations
- Interactive dashboard with sliders to adjust portfolio weights
- Supports multiple assets via CSV upload

---

## Installation

```bash

git clone https://github.com/Mahyar-Mirzaei/portfolio-optimization-dashboard.git
cd portfolio-optimization-dashboard

# 2. (Optional) Create a virtual environment
python -m venv venv
# Activate the environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Install required packages
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

## CSV File Format

The CSV should have dates as rows and asset prices as columns.

### Notes:

The Date column must be in YYYY-MM-DD format.

Asset columns contain daily closing prices.

A sample file data.csv is included for testing.



