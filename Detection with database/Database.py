from pymongo import MongoClient


class Database:
    """Class to handle database operations

    Attributes:
        client: MongoDB client
        db: Database object
        collection: Collection object
    """

    def __init__(self, db_name, collection_name):
        """Initialize the database connection

        Args:
            db_name: Name of the database
            collection_name: Name of the collection

        Returns:
            collection: The collection object

        """
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def get_collection(self):
        """Get the collection object

        Returns:
            collection: The collection object
        """

        return self.collection

    def insert_data(self, data):
        """Insert data into the collection

        Args:
            data: Data to be inserted
        """
        self.collection.insert_one(data)

    def find_data(self, query):
        """Find data in the collection

        Args:
            query: Query to search for

        Returns:
            Data that matches the query
        """
        return self.collection.find(query)
