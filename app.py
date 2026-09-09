from flask import Flask, jsonify, request

app = Flask(__name__)


items_database = [{"id": 1, "name": "Hackathon Project Plan"}]

# 1. READ Route (Lets anyone see the data)
@app.route('/api/items', methods=['GET'])
def get_items():
    return jsonify(items_database)

# 2. CREATE Route (Lets anyone add new data)
@app.route('/api/items', methods=['POST'])
def add_item():
    data = request.get_json()
    new_item = {
        "id": len(items_database) + 1,
        "name": data.get("name", "Unnamed Item")
    }                                                                                               
    items_database.append(new_item)
    return jsonify(new_item), 201

# 3. UPDATE Route (Modify an existing item)
@app.route('/api/items/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    data = request.get_json()
    for item in items_database:
        if item["id"] == item_id:
            item["name"] = data.get("name", item["name"])
            return jsonify(item)
    return jsonify({"error": "Item not found"}), 404

# 4. DELETE Route (Remove an item)
@app.route('/api/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    global items_database
    items_database = [item for item in items_database if item["id"] != item_id]
    return jsonify({"message": "Item deleted successfully"}), 200
# GET a single item by its ID
@app.route('/api/items/<int:item_id>', methods=['GET'])
def get_single_item(item_id):
    for item in items_database:
        if item["id"] == item_id:
            return jsonify(item)
    return jsonify({"error": "Item not found"}), 404

if __name__ == '__main__':
    # This runs your server live
    app.run(port=3000, debug=True)