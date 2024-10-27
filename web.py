# Import libraries
import io
import flask
import base64
import flask_cors

from cnn import CNN
from PIL import Image
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
model.load_state_dict(load("mnist_cnn.pth", weights_only=True))
model.eval()


# Initialize the Flask app
app = flask.Flask(__name__)
flask_cors.CORS(app)
current_image = None


def preprocess_image(image):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    image = Image.open(io.BytesIO(image))

    # convert image to base64 and store it in the global variable
    global current_image
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    current_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return preprocess(image).unsqueeze(0)


# Define a route for the prediction
@app.route("/predict", methods=["POST"])
def predict():
    image = flask.request.files.get("image")
    if image is None:
        return flask.redirect(flask.url_for("index"))

    image = preprocess_image(image.read()).to("cuda")
    prediction = model(image).argmax().item()

    return flask.redirect(flask.url_for("index", prediction=prediction))


@app.route("/")
def index():
    prediction = flask.request.args.get("prediction") or ""
    global current_image

    return f"""
    <!doctype html>
    <html lang=en>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.tailwindcss.com"></script>
            <title>CNN MNIST</title>
        </head>
        <body class="bg-black text-white min-h-dvh grid place-items-center">
            <main class="container mx-auto flex flex-col gap-4 max-w-screen-md">
                <h1 class="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl">Upload new File for Prediction</h1>

                <form method=post enctype=multipart/form-data action=predict class="flex gap-4 items-center">
                    <input type=file name=image class="flex h-10 w-full rounded-md border border-gray-500 bg-black px-3 py-2 text-sm ring-offset-black file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-white placeholder:text-gray-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50">
                    <button class="h-10 px-4 py-2 inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium bg-white text-black hover:bg-white/90 ring-offset-black transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0">
                        Upload
                    </button>
                </form>

                <div class="grid grid-cols-2 gap-4 place-items-center">
                    {current_image and f"""
                    <img src="data:image/png;base64,{current_image}" class="border border-white w-full rounded-lg" alt="Current Image">
                    """ or ""}
                    <h2 class="scroll-m-20 pb-2 text-3xl font-semibold tracking-tight first:mt-0 {current_image and "" or "col-start-2"}">Prediction: {prediction}</h2>
                </div>
            </main>
        </body>
    </html>
"""


def main():
    app.run()


if __name__ == "__main__":
    main()
