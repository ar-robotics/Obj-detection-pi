import json
import os


class JsonDatabase:
    def __init__(self, db_file):
        self.db_file = db_file
        if not os.path.exists(self.db_file):
            with open(self.db_file, "w") as f:
                json.dump({}, f)

    def _load_data(self):
        with open(self.db_file, "r") as f:
            return json.load(f)

    def _write_data(self, data):
        with open(self.db_file, "w") as f:
            json.dump(data, f, indent=4)

    def find_one(self, query):
        """Find a single document in the database."""
        data = self._load_data()
        for item in data.get(query.get("class"), []):
            if all(item.get(k) == v for k, v in query.items()):
                return item

    def find(self, query):
        data = self._load_data()

        return [
            item
            for item in data.get(query.get("class"), [])
            if all(item.get(k) == v for k, v in query.items())
        ]

    def insert_one(self, object_type, object_data):
        data = self._load_data()
        if object_type not in data:
            data[object_type] = []
        data[object_type].append(object_data)
        self._write_data(data)

    def update_one(self, query, update_data):
        data = self._load_data()
        items = data.get(query.get("type"), [])
        for i, item in enumerate(items):
            if all(item.get(k) == v for k, v in query.items()):
                items[i].update(update_data)
                self._write_data(data)
                return True
        return False

    def delete_one(self, query):
        data = self._load_data()
        items = data.get(query.get("type"), [])
        data[query.get("type")] = [
            item
            for item in items
            if not all(item.get(k) == v for k, v in query.items())
        ]
        self._write_data(data)

    def insert_many(self, items):
        """Insert multiple items into the database."""
        data = self._load_data()
        for item in items:
            object_type = item.get("type", "unknown")
            if object_type not in data:
                print(f"Creating new object type: {object_type}")
                data[object_type] = []
            data[object_type].append(item)
        self._write_data(data)


# db = JsonDatabase("data.json")

# print(db.find_one({"class": "person"}))
