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
        for item in data.get(query.get("type", []), []):
            if all(item.get(k) == v for k, v in query.items()):
                print(item)
                return item
        return None

    def find(self, query):
        data = self._load_data()
        return [
            item
            for item in data.get(query.get("type"), [])
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
            object_type = item.get("type", "unknown")  # Keep 'type' inside the item
            if object_type not in data:
                print(f"Creating new object type: {object_type}")
                data[object_type] = []
            data[object_type].append(item)
        self._write_data(data)


# Example usage:
db = JsonDatabase("db.json")


data = [
    {
        "type": "person",
        "name": "John Doe",
        "age": 30,
        "occupation": "Software Developer",
    },
    {
        "type": "bicycle",
        "brand": "Trek",
        "model": "Emonda",
        "color": "Red",
        "frameSize": "56cm",
    },
    {
        "type": "car",
        "brand": "Mercedes",
        "model": "Emonda",
        "color": "Red",
        "year": 2019,
    },
    {"type": "motorcycle", "brand": "Harley Davidson", "model": "v2"},
    {"type": "airplane", "brand": "Boeing", "color": "White"},
    {"type": "bus", "brand": "Greyhound", "color": "White"},
    {"type": "train", "brand": "Amtrak", "color": "White"},
    {"type": "truck", "brand": "Ford", "color": "White"},
    {"type": "boat", "brand": "Yamaha", "color": "White"},
    {"type": "traffic light", "color": "Red"},
    {"type": "fire hydrant", "color": "Red"},
    {"type": "street sign", "color": "Red-Yellow-Green"},
    {"type": "stop sign", "color": "Red"},
    {
        "type": "parking meter",
        "color": "Red",
        "weight": "50",
        "location": "Downtown",
        "timeLimit": "2 hours",
    },
    {
        "type": "bench",
        "color": "Red",
        "material": "Wood",
        "location": "Central Park",
        "length": "5 feet",
    },
    {
        "type": "bird",
        "color": "Red",
        "species": "Cardinal",
        "habitat": "Woodlands",
        "diet": "Seeds",
    },
    {
        "type": "cat",
        "color": "Red",
        "breed": "Orange Tabby",
        "age": "3 years",
        "name": "Garfield",
    },
    {
        "type": "dog",
        "color": "Red",
        "breed": "Irish Setter",
        "age": "5 years",
        "name": "Rusty",
    },
    {
        "type": "horse",
        "color": "Red",
        "breed": "Chestnut",
        "age": "7 years",
        "name": "Copper",
    },
    {
        "type": "sheep",
        "color": "Red",
        "breed": "Merino",
        "age": "4 years",
        "woolQuality": "High",
    },
    {
        "type": "cow",
        "color": "Red",
        "breed": "Aberdeen Angus",
        "age": "5 years",
        "milkProduction": "Low",
    },
    {
        "type": "elephant",
        "color": "Red",
        "species": "African",
        "age": "20 years",
        "name": "Ruby",
    },
    {
        "type": "bear",
        "color": "Red",
        "species": "Brown Bear",
        "habitat": "Forest",
        "diet": "Omnivore",
    },
    {
        "type": "zebra",
        "color": "Red",
        "species": "Plains Zebra",
        "habitat": "Savanna",
        "stripes": "Unique",
    },
    {
        "type": "giraffe",
        "color": "Red",
        "species": "Masai Giraffe",
        "habitat": "Savanna",
        "height": "16 feet",
    },
    {
        "type": "hat",
        "color": "Red",
        "material": "Wool",
        "size": "Medium",
        "style": "Beanie",
    },
    {
        "type": "backpack",
        "color": "Red",
        "material": "Nylon",
        "size": "20L",
        "brand": "North Face",
    },
    {
        "type": "umbrella",
        "color": "Red",
        "size": "Large",
        "brand": "Totes",
        "waterproof": "Yes",
    },
    {
        "type": "shoe",
        "color": "Red",
        "size": "10",
        "brand": "Nike",
        "style": "Sneakers",
    },
    {
        "type": "eye glasses",
        "color": "Red",
        "brand": "Ray-Ban",
        "frameMaterial": "Metal",
        "lensType": "Prescription",
    },
    {
        "type": "handbag",
        "color": "Red",
        "material": "Leather",
        "brand": "Gucci",
        "size": "Medium",
    },
    {
        "type": "tie",
        "color": "Red",
        "material": "Silk",
        "length": "Standard",
        "brand": "Hugo Boss",
    },
    {
        "type": "suitcase",
        "color": "Red",
        "material": "Polycarbonate",
        "size": "28 inch",
        "brand": "Samsonite",
    },
    {
        "type": "frisbee",
        "color": "Red",
        "diameter": "10 inches",
        "material": "Plastic",
        "brand": "Wham-O",
    },
    {
        "type": "skis",
        "color": "Red",
        "length": "170 cm",
        "brand": "Rossignol",
        "type-1": "All Mountain",
    },
    {
        "type": "snowboard",
        "color": "Red",
        "length": "155 cm",
        "brand": "Burton",
        "type-1": "Freestyle",
    },
    {
        "type": "sports ball",
        "color": "Red",
        "exact type": "Soccer Ball",
        "size": "5",
        "brand": "Adidas",
    },
    {
        "type": "kite",
        "color": "Red",
        "size": "Large",
        "material": "Nylon",
        "style": "Delta",
    },
    {
        "type": "baseball bat",
        "color": "Red",
        "material": "Aluminum",
        "length": "34 inches",
        "brand": "Easton",
    },
    {
        "type": "baseball glove",
        "color": "Red",
        "size": "12 inches",
        "material": "Leather",
        "brand": "Wilson",
    },
    {
        "type": "skateboard",
        "color": "Red",
        "length": "32 inches",
        "brand": "Santa Cruz",
        "exact type": "Street",
    },
    {
        "type": "surfboard",
        "color": "Red",
        "length": "6 feet",
        "exact type": "Shortboard",
        "brand": "Quiksilver",
    },
    {
        "type": "tennis racket",
        "color": "Red",
        "brand": "Wilson",
        "model": "Pro Staff",
        "gripSize": "4 3/8",
    },
    {
        "type": "bottle",
        "color": "Red",
        "material": "Stainless Steel",
        "capacity": "1L",
        "brand": "Hydro Flask",
    },
    {"type": "plate", "color": "Red"},
    {"type": "wine glass", "color": "Red", "year bought": "2012"},
    {"type": "cup", "color": "Red"},
    {"type": "fork", "color": "Red"},
    {"type": "knife", "color": "Red"},
    {"type": "spoon", "color": "Grey", "brand": "Amtrak"},
    {"type": "bowl", "color": "Red"},
    {"type": "banana", "color": "Yellow"},
    {"type": "apple", "color": "Red"},
    {
        "type": "sandwich",
        "toppings": ["Cheese", "Lettuce", "Tomato", "Ham"],
        "size": "Large",
    },
    {"type": "orange", "color": "Orange"},
    {"type": "broccoli", "color": "Green"},
    {"type": "carrot", "color": "Orange"},
    {"type": "hot dog", "toppings": ["Ketchup", "Mustard", "Relish"], "size": "Large"},
    {
        "type": "pizza",
        "toppings": ["Cheese", "Pepperoni", "Mushrooms"],
        "size": "Large",
        "crust": "Thin",
    },
    {
        "type": "donut",
        "toppings": ["Chocolate", "Glazed", "Sprinkles"],
        "size": "Large",
    },
    {
        "type": "cake",
        "toppings": ["Chocolate", "Vanilla", "Strawberry"],
        "size": "Large",
    },
    {
        "type": "chair",
        "color": "Red",
        "material": "Wood",
        "style": "Modern",
        "brand": "IKEA",
    },
    {
        "type": "couch",
        "color": "Red",
        "material": "Leather",
        "style": "Sectional",
        "brand": "West Elm",
    },
    {
        "type": "potted plant",
        "color": "Red",
        "species": "Red Aglaonema",
        "potMaterial": "Ceramic",
        "brand": "Costa Farms",
    },
    {
        "type": "bed",
        "color": "Red",
        "size": "Queen",
        "material": "Wood",
        "brand": "Zinus",
    },
    {
        "type": "mirror",
        "color": "Red",
        "shape": "Oval",
        "frameMaterial": "Metal",
        "brand": "Umbra",
    },
    {
        "type": "dining table",
        "color": "Red",
        "material": "Glass",
        "shape": "Rectangle",
        "brand": "CB2",
    },
    {
        "type": "window",
        "color": "Red",
        "type-1": "Sliding",
        "frameMaterial": "Vinyl",
        "brand": "Pella",
    },
    {
        "type": "desk",
        "color": "Red",
        "material": "Metal",
        "style": "Adjustable",
        "brand": "Fully",
    },
    {
        "type": "toilet",
        "color": "White",
        "brand": "Porsgrunn",
        "material": "Ceramic",
        "flushType": "Dual",
    },
    {
        "type": "door",
        "color": "Red",
        "material": "Wood",
        "type-1": "Panel",
        "brand": "Masonite",
    },
    {
        "type": "tv",
        "color": "Red",
        "brand": "Samsung",
        "size": "55 inch",
        "screentype": "LED",
    },
    {
        "type": "laptop",
        "color": "Red",
        "brand": "Dell",
        "model": "Inspiron",
        "screenSize": "15 inch",
    },
    {
        "type": "mouse",
        "color": "Red",
        "brand": "Logitech",
        "type-1": "Wireless",
        "model": "M330",
    },
    {
        "type": "remote",
        "color": "Red",
        "brand": "Sony",
        "type-1": "Universal",
        "compatibleDevices": "TV, Blu-Ray",
    },
    {
        "type": "keyboard",
        "color": "Red",
        "brand": "Corsair",
        "surface": "Mechanical",
        "model": "K70",
    },
    {
        "type": "cell phone",
        "color": "Red",
        "brand": "Apple",
        "model": "iPhone 12",
        "storage": "128GB",
    },
    {
        "type": "microwave",
        "color": "Red",
        "brand": "Panasonic",
        "type-1": "Countertop",
        "capacity": "2.2 cu ft",
    },
    {
        "type": "oven",
        "color": "Red",
        "brand": "KitchenAid",
        "type-1": "Convection",
        "capacity": "5.8 cu ft",
    },
    {
        "type": "toaster",
        "color": "Red",
        "brand": "Smeg",
        "slots": "2",
        "settings": "6 Browning Levels",
    },
    {
        "type": "sink",
        "color": "Red",
        "material": "Stainless Steel",
        "type-1": "Undermount",
        "brand": "Kohler",
    },
    {
        "type": "refrigerator",
        "color": "Red",
        "brand": "LG",
        "type-1": "French Door",
        "capacity": "22 cu ft",
    },
    {
        "type": "blender",
        "color": "Red",
        "brand": "Vitamix",
        "model": "5200",
        "capacity": "64 oz",
    },
    {
        "type": "book",
        "color": "Red",
        "title": "The Red Book",
        "author": "C.G. Jung",
        "genre": "Psychology",
    },
    {
        "type": "clock",
        "color": "Red",
        "type-1": "Wall Clock",
        "brand": "Ikea",
        "diameter": "10 inches",
    },
    {
        "type": "vase",
        "color": "Red",
        "material": "Glass",
        "height": "12 inches",
        "brand": "Crate & Barrel",
    },
    {
        "type": "scissors",
        "color": "Red",
        "brand": "Fiskars",
        "type-1": "General Purpose",
        "length": "8 inches",
    },
    {
        "type": "teddy bear",
        "color": "Red",
        "material": "Plush",
        "brand": "Steiff",
        "height": "16 inches",
    },
    {
        "type": "hair drier",
        "color": "Red",
        "brand": "Dyson",
        "model": "Supersonic",
        "technology": "Ionic",
    },
    {
        "type": "toothbrush",
        "color": "Red",
        "brand": "Oral-B",
        "type-1": "Electric",
        "model": "Pro 100",
    },
    {
        "type": "hair brush",
        "color": "Red",
        "brand": "Mason Pearson",
        "type-1": "Boar Bristle",
        "size": "Large",
    },
]
