# Import libraries
import cv2
import flask
import flask_cors
import numpy as np

from cnn import CNN
from torch import nn, load
from torchvision import transforms

# Load the model
model = CNN(
    [
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Flatten(),
        nn.Linear(64 * 14 * 14, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.Softmax(dim=1),
    ]
)
model.load_state_dict(load("./mnist_cnn.pth", weights_only=True))
model.eval()


# Initialize the Flask app
app = flask.Flask(__name__)
flask_cors.CORS(app)


def preprocess_image(image):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image = cv2.resize(image, (28, 28))

    return preprocess(image).unsqueeze(0).to("cuda")


@app.route("/predict", methods=["POST"])
def predict():
    image = flask.request.files.get("image")
    image = preprocess_image(image)

    prediction = model(image).argmax().item()

    return flask.jsonify({"prediction": prediction})


@app.route("/")
def index():
    return flask.render_template("index.html")


if __name__ == "__main__":
    app.run()
