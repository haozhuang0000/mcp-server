from src.config import (
    MYSQL_HOST,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
    MYSQL_PORT,
)
import pymysql
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

class MySQLHandler:

    def __init__(self, user, password, host, database, port=3306, echo=False):
        """
        Initialize MySQL handler with SQLAlchemy.
        """
        url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        self.engine = create_engine(url, echo=echo, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)
        self.session = None

    def fetch_df(self, query, params=None):
        """
        Fetch query results directly into a Pandas DataFrame.
        """
        try:
            df = pd.read_sql(text(query), self.engine, params=params)
            return df
        except Exception as e:
            print(f"❌ Error fetching DataFrame: {e}")
            raise