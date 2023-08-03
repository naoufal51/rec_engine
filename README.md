# Recommendations with IBM
## Description
In this project, we create a recommendation system based on data collected from IBM Watson Studio Platform.
The data represents the interaction between the platform users and the hosted articles.
The recommendation system provides recommendations of articles using different methods:
1. Rank-Based Recommendations
2. User-User Based Collaborative Filtering
3. Matrix Factorization

https://github.com/naoufal51/rec_engine/assets/15954923/dc723add-9910-4274-ad56-a63ac84a2c1e

## Live App
**Click on the following link to access the web app:**

[Recommendation System webapp](https://recengine-uypcoopcphebsynchvybxa.streamlit.app/)

## Getting Started
### Dependencies
- Python 3.9
- NumPy
- Pandas
- Scikit-Learn
- nbconvert
- streamlit
- Plotly

### Local Installation
- Clone the repository
```bash
git clone https://github.com/naoufal51/rec_engine.git
```
- Create a virtual environment
```bash
python3 -m venv .venv
```
- Activate the virtual environment
```sh
source venv/bin/activate
```
- Install the dependencies
```bash
pip install -r requirements.txt
```
- Run the application
```bash
streamlit run app.py
```

## Acknowledgements
[IBM Watson Platform](https://www.ibm.com/products/watson-studio) for providing the data.
