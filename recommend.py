import pandas as pd
import numpy as np

class Recommender:
    """
    Recommender class.
    Used to provide article recommendations to users based on collaborative filtering.
    """
    def __init__(self, user_content_path=None, content_path=None):
        self.user_content_path = 'data/user-item-interactions.csv' if user_content_path is None else user_content_path
        self.content_path = 'data/articles_community.csv' if content_path is None else content_path

        self.df = self._load_clean_data(self.user_content_path)
        self.df_content = self._load_clean_data(self.content_path)

        
        email_encoded = self._email_mapper()
        del self.df['email']
        self.df['user_id'] = email_encoded

        self.user_item = self._create_user_item_matrix(self.df)

    def _load_clean_data(self, file_path):
        """
        Load and clean the data.
        
        Args:
            file_path (str): Path to the data.
        
        Returns:
            df (pd.DataFrame): Cleaned dataframe
        
        """
        try:
            df = pd.read_csv(file_path)
            if 'Unnamed: 0' in df.columns:
                del df['Unnamed: 0']
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f'{file_path} not found')
        
    def _email_mapper(self):
        """
        Return a list of coded emails.
        Apply a mapping between user emails to a new user_id column.

        Args:
            None
        
        Returns:
            email_encoded (list): List of user ids
        
        """
        coded_dict = dict()
        cter = 1
        email_encoded = []

        for val in self.df['email']:
            if val not in coded_dict:
                coded_dict[val] = cter
                cter+=1

            email_encoded.append(coded_dict[val])
        return email_encoded

    def _create_user_item_matrix(self, df):
        """
        Return matric with user ids as rows and article ids as columns.
        Each cell represents if the user interacted with the article.
        1 = interacted, 0 = not interacted.
        
        Args:
            df (pd.DataFrame): DataFrame with article_id, title, user_id columns
        
        Returns:
            user_item (pd.DataFrame): Matrix with user ids as rows and article ids as columns.
        
        """
        user_item = df.drop_duplicates()
        user_item = user_item.set_index('user_id')
        user_item = pd.get_dummies(user_item['article_id']).groupby('user_id').sum().clip(upper=1)
        user_item = user_item.astype(int)
        return user_item

    def get_top_article_ids(self, n):
        """
        Return the top n article ids.
        
        Args:
            n (int): Number of top articles to return
        
        Returns:
            top_articles (list): List of top n article ids
        
        """
        top_articles = list(self.df['article_id'].value_counts().iloc[:n].index)
        top_articles = [str(x) for x in top_articles]
        return top_articles

    def get_article_names(self, article_ids):
        """
        Return a list of the names of the articles from article ids.
        
        Args:
            article_ids (list): Article ids
        
        Returns:
            article_names (list): Article names
        
        
        """
        article_names = self.df.copy().drop_duplicates(subset=['article_id']).set_index('article_id')    
        article_names = [article_names.loc[float(a_id)]['title'] for a_id in article_ids]
        return article_names

    def get_user_articles(self, user_id):
        """
        Gets a list of article_ids and articles_names that the user interacted with.

        Args:
            user_id (int): User id
        
        Returns:
            article_ids (list): List of article ids already seen by the user.
            article_names (list): List of article names corresponding to the article_ids.
        """
        row_user = self.user_item.loc[user_id]
        article_ids = row_user[row_user==1].index.tolist()
        article_ids = [str(x) for x in article_ids]
        article_names = self.get_article_names(article_ids)
        return article_ids, article_names

    def get_top_sorted_users(self, user_id):
        """
        Get most similar users to the user_id.
        The output neighbors_df is sorted by the similarity and the number of interactions (descending order).

        Args:
            user_id (int): User id
        
        Returns:
            neighbors_df (pd.DataFrame): DataFrame with neighbor_id and similarity columns.
        """
        most_similar_users = self.user_item.drop(user_id).dot(self.user_item.loc[user_id])
        most_similar_users = most_similar_users.to_frame(name='similarity')
        interactions = self.df.groupby('user_id')['article_id'].count().to_frame(name='num_interactions')
        neighbors_df = most_similar_users.merge(interactions, on='user_id')
        neighbors_df = neighbors_df.sort_values(by=['similarity','num_interactions'], ascending=[False, False])
        neighbors_df.reset_index(inplace=True)
        neighbors_df.rename(columns={'user_id': 'neighbor_id'}, inplace=True)
        return neighbors_df

    def user_user_recs_part2(self, user_id, m=10):
        """
        Using collaborative filtering, we get m recommendations for a given user_id.
        The recommendations are based on:
            1. The most similar users to the user_id.
            2. The articles that the user has not interacted with.
            4. The users that have the most total article interactions first.
            3. The articles with the most total interactions first.
        
        Args:
            user_id (int): User id
            m (int): Number of recommendations to return
        
        Returns:
            recs (list): List of the recommended m article ids
            rec_names (list): List of the recommended m article names
        
        """
        user_articles, _ = self.get_user_articles(user_id)
        user_articles = set(user_articles)

        top_similar_users = self.get_top_sorted_users(user_id)
        top_articles = self.get_top_article_ids(len(self.df))

        recs = []

        for user in top_similar_users['neighbor_id']:
            rec_user_articles, _  = self.get_user_articles(user)
            new_rec = list(set(rec_user_articles) - user_articles - set(recs))
            new_rec_sorted = [art for art in top_articles if art in new_rec]
            empty_slots = m - len(recs)
            if empty_slots > 0:
                recs.extend(new_rec_sorted[:empty_slots])
            else:
                break

        rec_names = self.get_article_names(recs)

        return recs, rec_names
