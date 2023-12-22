from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/greet', methods=['POST'])
def greet():
    data = request.json
    name = data.get('name', 'Guest')  # Default to 'Guest' if no name is provided
    return jsonify(message=f"Hello {name}")

if __name__ == '__main__':
    app.run(debug=True)
