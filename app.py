import streamlit as st
from recommend import Recommender
import pandas as pd
import plotly.graph_objects as go

@st.cache_data
def load_model():
    model = Recommender(user_content_path='data/user-item-interactions.csv', content_path='data/articles_community.csv')
    return model

def main():
    st.title('IBM Article Recommender System')
    st.markdown("<h3 style='text-align: center; color: white;'>Welcome to our article recommender system.</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: white;'>Enter a user id to get personalized article recommendations.</h4>", unsafe_allow_html=True)
    st.write("We use Collaborative Filtering to generate recommendations. If no user is specified, we recommend the most popular articles.")
    st.image('images/screen-shot-2018-09-17-at-3.40.30-pm.png', caption='Dashboard for Articles on the IBM Watson Platform. source: Udacity')

    model = load_model()

    st.sidebar.header("User Input")
    user_id = st.sidebar.number_input("Enter a user id (integer):", min_value=1, step=1, format="%i")
    num_recs = st.sidebar.slider("Number of Recommendations:", min_value=1, max_value=20, value=10, step=1)

    if st.sidebar.button('Recommend'):
        if user_id:
            try:
                recs, rec_names = model.user_user_recs_part2(user_id, m=num_recs)
                st.markdown("<h4 style='text-align: center; color: white;'>Recommended Articles:</h4>", unsafe_allow_html=True)
                data = {'Article Ids': recs, 'Article Names': rec_names}
                df = pd.DataFrame(data)
                st.table(df)
            except KeyError:
                st.sidebar.write("User id not found. Please try a another id.")
        else:
            st.sidebar.write("Please enter a user id.")
    else:
        st.markdown("<h4 style='text-align: center; color: white;'>Most Popular Articles:</h4>", unsafe_allow_html=True)
        top_articles = model.get_top_article_ids(num_recs)
        top_article_names = model.get_article_names(top_articles)
        data = {'Article Ids': top_articles, 'Article Names': top_article_names}
        df = pd.DataFrame(data)
        st.table(df)
        
    st.divider()
    st.markdown("<h4 style='text-align: center; color: white;'>Histogram of User-Article Interactions</h4>", unsafe_allow_html=True)
    hist_values = model.df['user_id'].value_counts().values
    fig = go.Figure(data=[go.Histogram(x=hist_values, nbinsx=20, marker_color='LightSkyBlue')])
    fig.update_layout(
        title={
            'text': "User-Article Interactions",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title_text='Number of Interactions',
        yaxis_title_text='Count',
        bargap=0.2,
        
        bargroupgap=0.1)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
