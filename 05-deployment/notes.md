# Machine Learning Zoomcamp 2026 — Module 05: Deployment

We'll study the entire lesson from the beginning to the end, in simple language, as if I'm teaching you in class. I'll explain what each concept means, why we need it, how the code works, and how the different pieces connect.

For now, we'll focus only on the Module 05 study lesson and workshop. We'll leave the homework, its questions, and its answers until you've understood the lesson.

## The big picture: what are we learning?

In Modules 1–4, you learned how to train models and evaluate their performance. But a trained model sitting in a Jupyter notebook is not yet a usable application.

Imagine you build a model that predicts whether a customer will buy a product. You now want a website to send customer information to your model and receive a prediction immediately.

That is where deployment comes in.

The official updated workshop covers these topics in order: model serialization, scikit-learn pipelines, converting a notebook into a training script, FastAPI, Pydantic validation, `uv`, Docker, and cloud deployment. We'll follow that order.&#x20;





## Lesson roadmap

### 01

Saving and loading a model

pickle and serialization

### 02

Scikit-learn Pipeline

Combining preprocessing and prediction

### 03

Notebook to Python script

Separating training from serving

### 04

HTTP and FastAPI

Building a web API

### 05

Serving predictions

Receiving input and returning probabilities

### 06

Testing the API

Using Swagger, curl and requests

### 07

Pydantic validation

Checking input and output

### 08

Environment management

uv, dependencies and lockfiles

### 09

Docker

Images, containers and Dockerfile

### 10

Cloud deployment

Deploying and testing a public service

# Part 1 — Saving and loading a machine learning model

## 1.1 Why do we need to save a model?

Suppose you trained a logistic regression model in Module 3.

```
model.fit(X_train, y_train)
```

The model has now learned parameters from your training data. While your Python session is running, the trained model exists in memory.

But what happens when you close Jupyter Notebook or shut down your computer?

The model object in memory is gone. You would need to train it again unless you saved it.

This is why we serialize the model.

Serialization means converting an object into a form that can be stored and restored later.

In this workshop, we use Python's `pickle` module.

## 1.2 Saving a model with pickle

```
import picklewith open("model.bin", "wb") as f:    pickle.dump(model, f)
```

Let's understand each line.

| Code                      | Simple explanation                        |
| ------------------------- | ----------------------------------------- |
| `import pickle`           | Imports Python's serialization module     |
| `open("model.bin", "wb")` | Opens a file for writing binary data      |
| `as f`                    | Gives the open file the variable name `f` |
| `pickle.dump(model, f)`   | Saves the model into that file            |

Two things are especially important:

- `"wb"` means write binary.
- `pickle.dump()` writes an object into a file.

The filename `model.bin` is not special. You could use another filename, but `.bin` is commonly used for binary artifacts.

## 1.3 Loading the model

Later, in a different Python process, you can restore the model:

```
import picklewith open("model.bin", "rb") as f:    model = pickle.load(f)
```

Here, `"rb"` means read binary, and `pickle.load()` restores the object.

Now you can use the model without retraining it:

```
predictions = model.predict(X_new)
```

The important idea is that training and prediction can happen at different times and in different programs.

## 1.4 Why saving only the classifier may not be enough

Remember the classification work from Module 3.

Your raw data might look like this:

```
customer = {    "gender": "female",    "contract": "month-to-month",    "tenure": 1,    "monthlycharges": 29.85}
```

A logistic regression model cannot directly perform its mathematical calculations on strings such as `"female"` and `"month-to-month"`.

Before prediction, these features must be transformed into numerical values.

The original workshop used two objects:

- `dv`: a `DictVectorizer` that converts feature dictionaries into numerical vectors.
- `model`: the trained classifier.

The prediction process therefore looks like this:

Raw customer data

Strings and numbers

DictVectorizer

Converts features into numerical vectors

Trained classifier

Calculates the prediction

Predicted probability

The original method saves both objects together:

```
with open("model.bin", "wb") as f:    pickle.dump((dv, model), f)
```

And restores both:

```
with open("model.bin", "rb") as f:    dv, model = pickle.load(f)
```

This works, but you have to remember to use the correct vectorizer with the correct model.

That brings us to the next topic.

Important security note: never load pickle files from untrusted sources. Deserializing a malicious pickle can execute code. Also, saved scikit-learn objects can depend on compatible Python and library versions.

# Part 2 — Scikit-learn Pipelines

## 2.1 What is a Pipeline?

A scikit-learn `Pipeline` combines several machine learning steps into one object.

Think of it as a production line in a factory:

1. The first station prepares the raw material.
2. The second station performs the main operation.
3. The finished product comes out at the end.

For our model:

1. `DictVectorizer` transforms the features.
2. `LogisticRegression` predicts the outcome.

Instead of managing these objects separately, we combine them.

## 2.2 Creating a Pipeline

```
from sklearn.pipeline import make_pipelinefrom sklearn.feature_extraction import DictVectorizerfrom sklearn.linear_model import LogisticRegressionpipeline = make_pipeline(    DictVectorizer(),    LogisticRegression(solver="liblinear"))
```

Let's break this down.

- `make_pipeline()` creates a pipeline.
- `DictVectorizer()` is the preprocessing step.
- `LogisticRegression(...)` is the prediction step.
- `solver="liblinear"` specifies the algorithm used to fit logistic regression.

This code creates the pipeline; it does not train it yet.

We train the complete pipeline:

```
pipeline.fit(train_dict, y_train)
```

Here:

- `train_dict` is a list of dictionaries containing the training features.
- `y_train` contains the corresponding target labels.

For example, the feature data might be structured like this:

```
train_dict = [    {"gender": "female", "tenure": 1},    {"gender": "male", "tenure": 24},    {"gender": "female", "tenure": 36}]y_train = [1, 0, 0]
```

These are illustrative values, not the full course dataset.

The pipeline learns the feature transformation and fits the classifier.

## 2.3 Making a prediction with the Pipeline

Suppose you have a new customer:

```
datapoint = {    "gender": "female",    "tenure": 1,    "monthlycharges": 29.85}
```

Use:

```
probability = pipeline.predict_proba([datapoint])[0, 1]
```

Notice the square brackets around `datapoint`.

`[datapoint]` represents a batch containing one observation. A `DictVectorizer` pipeline expects an iterable of feature dictionaries, even if you have only one observation.

Let's understand the result selection:

```
pipeline.predict_proba([datapoint])
```

This returns probabilities for the classes, for example:

```
[[0.27, 0.73]]
```

The values are illustrative. The actual output depends on the fitted model.

The expression:

```
[0, 1]
```

selects:

- `0`: the first observation.
- `1`: the probability for class `1`.

Therefore, `0.73` would mean the model estimates a 73% probability for class `1`.

## 2.4 Save the complete Pipeline

Now you can save just one object:

```
with open("model.bin", "wb") as f:    pickle.dump(pipeline, f)
```

Later:

```
with open("model.bin", "rb") as f:    pipeline = pickle.load(f)
```

Then predict:

```
probability = pipeline.predict_proba([datapoint])[0, 1]
```

You don't need to manually call `dv.transform()` anymore.

Why this matters: the preprocessing learned during training is reused when the model receives new data. This helps prevent differences between training-time and prediction-time feature handling.

# Part 3 — Turning a notebook into a training script

## 3.1 Why move away from notebooks?

Jupyter Notebook is excellent for:

- Experimenting with data.
- Visualizing results.
- Testing ideas.
- Evaluating models.
- Exploring errors.

But imagine that you need to retrain a model every week. Opening a notebook and manually running cells is inconvenient and can lead to mistakes.

A Python script allows you to run the same workflow consistently from a terminal.

The workshop demonstrates converting the notebook to a Python script using `nbconvert`.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



```
jupyter nbconvert --to=script workshop-uv-fastapi.ipynb
```

This creates a Python script from the notebook.

In the workshop, that generated script is then renamed to `train.py`.

The conversion is only a starting point. You should inspect the resulting code, remove notebook-specific commands where necessary, and organize it into functions.

## 3.2 Structure the training script

A clean training script separates the main tasks:

```
def load_data():    # Read and prepare the dataset    passdef train_model(df):    # Create and fit the pipeline    passdef save_model(pipeline, filename):    # Save the fitted pipeline    passif __name__ == "__main__":    df = load_data()    pipeline = train_model(df)    save_model(pipeline, "model.bin")    print("Model saved")
```

This is a structural example, not a complete runnable training program. The `pass` statements must be replaced with real code.

Let's examine the three functions.

### `load_data()`

Loads the dataset and performs any required cleaning.

### `train_model(df)`

Creates the feature representation, builds the pipeline, and fits the model.

### `save_model(pipeline, filename)`

Serializes the fitted pipeline to a file.

The final section runs the workflow when you execute the file directly:

```
python train.py
```

## 3.3 Why the `if __name__ == "__main__":` statement?

This Python pattern tells the program to run the training workflow when the file is executed directly.

It also allows another Python file to import functions from `train.py` without automatically starting training.

This becomes useful as your project grows.

## 3.4 Separate training and serving

This is one of the most important concepts in the module.

| Training program     | Prediction service      |
| -------------------- | ----------------------- |
| Loads training data  | Loads a saved model     |
| Learns parameters    | Uses learned parameters |
| May take a long time | Should respond promptly |
| Produces `model.bin` | Reads `model.bin`       |

You usually train the model first and then run the prediction service separately.

The API should not retrain the model whenever a customer sends a request.

# Part 4 — Understanding HTTP and APIs

Before building the prediction service, we need to understand how programs communicate.

## 4.1 What is an API?

API stands for Application Programming Interface.

In this context, an API lets one program ask another program to perform an operation.

Imagine this situation:

- A website collects a customer's details.
- The website sends those details to a Python application.
- The Python application uses the model.
- The application returns a prediction.
- The website displays the result.

The website doesn't need to know how logistic regression works. It only needs to know how to send a valid request and interpret the response.

## 4.2 What is HTTP?

HTTP is a communication protocol commonly used by web applications.

An HTTP request includes elements such as:

- A method.
- A URL path.
- Headers.
- Sometimes a body containing data.

Two methods are particularly important here.

| Method | Typical use                         |
| ------ | ----------------------------------- |
| `GET`  | Retrieve information                |
| `POST` | Submit data or request an operation |

Our application will have two endpoints:

- `GET /ping` — a simple test endpoint.
- `POST /predict` — accepts customer information and returns a prediction.

## 4.3 What is JSON?

JSON is a format for exchanging structured data.

Example:

```
{
  "gender": "female",
  "tenure": 1,
  "monthlycharges": 29.85
}
```

This represents an object containing three fields.

The keys identify the features, and the values contain their data.

JSON is useful because applications written in different programming languages can exchange information using the same format.

For example, a JavaScript website can send JSON to a Python API.

# Part 5 — FastAPI and Uvicorn

The updated workshop replaces the older Flask-based example with FastAPI.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 5.1 What is FastAPI?

FastAPI is a Python framework for building APIs.

It lets you define:

- Which URLs the application supports.
- Which HTTP methods those URLs accept.
- What data an endpoint expects.
- What information it returns.

## 5.2 What is Uvicorn?

Uvicorn is the server that runs the FastAPI application.

Think of the relationship this way:

- FastAPI defines the application's behavior.
- Uvicorn runs the application and accepts HTTP requests.

Installing FastAPI alone doesn't automatically mean that a web server is running.

## 5.3 Build your first FastAPI application

Create a file called `ping.py`.

```
from fastapi import FastAPIapp = FastAPI(title="ping")@app.get("/ping")def ping():    return "PONG"
```

Let's understand the code.

### `app = FastAPI(...)`

Creates the application object.

### `@app.get("/ping")`

This is a decorator that registers the function below it as the handler for GET requests to `/ping`.

### `def ping():`

Defines the Python function that runs when the endpoint is called.

### `return "PONG"`

Returns the response.

The complete interaction looks like this:

```
Browser or client
       |
       | GET /ping
       v
    Uvicorn
       |
       v
    FastAPI
       |
       v
  ping() function
       |
       v
    "PONG"
```

## 5.4 Start the server

From the terminal in the directory containing `ping.py`, run:

```
uvicorn ping:app --host 0.0.0.0 --port 9696 --reload
```

Each part matters.

| Part             | Meaning                                 |
| ---------------- | --------------------------------------- |
| `uvicorn`        | Starts the web server                   |
| `ping:app`       | The `app` object inside `ping.py`       |
| `--host 0.0.0.0` | Listen on all network interfaces        |
| `--port 9696`    | Listen on port 9696                     |
| `--reload`       | Restart automatically when code changes |

For local development, `--reload` is convenient. It is generally not needed in a production deployment.

You can also start Uvicorn from Python, but the command-line approach is the main pattern shown in the workshop.

## 5.5 Test your first endpoint

Open this URL in your browser:

[http://127.0.0.1:9696/ping](http://127.0.0.1:9696/ping?utm_source=chatgpt.com)

You should receive:

```
PONG
```

The address `127.0.0.1` means your own computer.

Now open:

[http://127.0.0.1:9696/docs](http://127.0.0.1:9696/docs?utm_source=chatgpt.com)

This is FastAPI's automatically generated interactive API documentation, based on the routes and schemas in your code.

You can use it to inspect endpoints and send requests without writing a separate client program.

# Part 6 — Turn the model into a prediction API

Now we combine everything we've learned.

We already have a saved pipeline. We want FastAPI to load it and use it to make predictions.

## 6.1 Load the model when the application starts

The core idea is:

```
import picklewith open("model.bin", "rb") as f:    pipeline = pickle.load(f)
```

This restores the trained pipeline.

In the workshop's serving script, the pipeline is loaded before the endpoint handles prediction requests. This avoids retraining the model or loading it afresh for every request.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 6.2 Create a prediction function

A prediction function separates model inference from the API endpoint.

```
def predict_single(customer):    probability = pipeline.predict_proba([customer])[0, 1]    return float(probability)
```

The `customer` argument is one dictionary of features.

The list `[customer]` tells the pipeline that we are predicting for one observation.

The `float()` conversion ensures that the returned probability is a regular Python float, suitable for JSON serialization.

## 6.3 Add the endpoint

```
from fastapi import FastAPIapp = FastAPI(title="customer-churn-prediction")@app.post("/predict")def predict(customer: dict):    probability = predict_single(customer)    return {        "churn_probability": probability,        "churn": bool(probability >= 0.5)    }
```

This is an illustrative minimal endpoint. Later, we'll replace the generic dictionary with a Pydantic model to validate the input more carefully.

Let's understand the endpoint.

### `@app.post("/predict")`

Defines a POST endpoint at `/predict`.

### `customer: dict`

Indicates that the request body should be represented as a dictionary. A more precise type annotation is needed for stronger validation and documentation.

### `predict_single(customer)`

Calls the trained model.

### `probability >= 0.5`

Converts the probability into a classification decision.

For example:

| Probability | Decision at threshold 0.5 |
| ----------- | ------------------------- |
| 0.91        | Churn predicted           |
| 0.68        | Churn predicted           |
| 0.43        | Churn not predicted       |
| 0.12        | Churn not predicted       |

These numbers are illustrative.

The threshold of `0.5` is a decision rule, not a universal requirement. In practice, you would select a threshold appropriate to the business problem and evaluation results.

## 6.4 What does the API return?

A response might look like this:

```
{
  "churn_probability": 0.73,
  "churn": true
}
```

Here, `0.73` is an illustrative probability. The `true` value is the model's classification under the selected threshold.

The API returns JSON rather than exposing the Python model object to the client.

## 6.5 Why specify the request type?

FastAPI needs to know how to interpret the incoming request body.

The workshop first introduces a dictionary type with `Dict[str, Any]` and then moves to a more explicit Pydantic schema.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



For a generic dictionary, you can write:

```
from typing import Any@app.post("/predict")def predict(customer: dict[str, Any]):    ...
```

This communicates that the request contains string keys and values of various types.

However, it doesn't fully describe which fields must exist, which values are allowed, or what numerical ranges make sense.

That's why we need Pydantic.

# Part 7 — Testing the prediction API

Testing means checking that your application accepts valid requests and produces the expected responses.

The workshop demonstrates several ways to test the service.

## 7.1 Method 1: FastAPI's interactive documentation

Start the server and open:

[http://127.0.0.1:9696/docs](http://127.0.0.1:9696/docs?utm_source=chatgpt.com)

Then:

1. Find `POST /predict`.
2. Expand the endpoint.
3. Click Try it out.
4. Enter the customer data.
5. Click Execute.
6. Inspect the response.

This is the easiest method when you're learning because you don't need to write a separate client.

## 7.2 Method 2: curl

`curl` is a command-line tool for sending HTTP requests.

A request to the prediction API looks like this:

```
curl -X POST "http://127.0.0.1:9696/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female",
    "tenure": 1,
    "monthlycharges": 29.85
  }'
```

This example only works if the request schema and model support those fields. The full workshop model expects additional customer features.

Understand the options:

- `-X POST` selects the HTTP method.
- `-H` supplies a request header.
- `Content-Type: application/json` tells the server that the body is JSON.
- `-d` supplies the request body.

The `curl` command returns the server's response.

## 7.3 Method 3: Python requests

Sometimes another Python program needs to use the prediction service.

The `requests` library makes this easy.

```
import requestsurl = "http://127.0.0.1:9696/predict"customer = {    "gender": "female",    "tenure": 1,    "monthlycharges": 29.85}response = requests.post(url, json=customer)print(response.status_code)print(response.json())
```

Again, the sample input must match the actual model's required schema.

Important details:

- `requests.post()` sends an HTTP POST request.
- `json=customer` serializes the Python dictionary into JSON and sets the appropriate content type.
- `response.status_code` shows the HTTP status.
- `response.json()` parses the JSON response into a Python object.

You can then use the result:

```
predictions = response.json()print(predictions["churn_probability"])if predictions["churn"]:    print("Customer is likely to churn")else:    print("Customer is not predicted to churn")
```

### Why is this important?

Your client doesn't need to load the model or know its internal implementation.

It only needs to send the correct data to the endpoint and interpret the returned JSON.

This is the fundamental idea behind model serving.

# Part 8 — Pydantic and input validation

This is one of the most important parts of the updated workshop.

## 8.1 What problem does validation solve?

Imagine that your model expects:

```
{
  "gender": "female",
  "tenure": 12,
  "monthlycharges": 50.0
}
```

But a client accidentally sends:

```
{
  "gender": "maybe",
  "tenure": "twelve",
  "monthlycharges": -500
}
```

What should the API do?

It should reject the invalid input rather than silently accept data that violates the expected schema.

Validation means checking whether incoming data satisfies defined rules before using it.

Pydantic allows us to define those rules.

## 8.2 Create a Pydantic model

The workshop uses `BaseModel`, `Literal`, and `Field` to define accepted customer data. Here is a smaller example demonstrating the same principles:

```
from typing import Literalfrom pydantic import BaseModel, Fieldclass Customer(BaseModel):    gender: Literal["male", "female"]    tenure: int = Field(ge=0)    monthlycharges: float = Field(ge=0)
```

Let's understand each part.

### `BaseModel`

This is the base class used to define a Pydantic data model.

### `Literal`

```
gender: Literal["male", "female"]
```

This restricts the value to one of the listed options.

The following is valid:

```
{"gender": "female", "tenure": 12, "monthlycharges": 50.0}
```

But `"gender": "maybe"` is not valid under this schema.

### `Field(ge=0)`

```
tenure: int = Field(ge=0)
```

The value must be an integer greater than or equal to zero.

Similarly:

```
monthlycharges: float = Field(ge=0)
```

The value must be a number greater than or equal to zero.

The validation rules must match the real dataset and business logic. They should not be invented just because they seem reasonable.

## 8.3 Required fields

In the example:

```
tenure: int
```

The field is required because it has no default value.

If the client leaves it out, Pydantic reports a validation error.

This is useful because missing features can cause a prediction to fail or produce an unintended result.

## 8.4 Define the output schema too

We can define the response structure:

```
class PredictResponse(BaseModel):    churn_probability: float    churn: bool
```

This describes the expected output.

Now we can use these models in the API:

```
@app.post("/predict", response_model=PredictResponse)def predict(customer: Customer) -> PredictResponse:    probability = predict_single(customer.model_dump())    return PredictResponse(        churn_probability=probability,        churn=probability >= 0.5    )
```

Notice the improvements:

- `customer: Customer` specifies the input schema.
- `model_dump()` converts the validated Pydantic object to a dictionary.
- `response_model=PredictResponse` specifies the response structure.
- The return type annotation communicates the expected result.

In Pydantic v2, use `model_dump()` instead of the older, deprecated `dict()` method.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 8.5 What if the client sends an extra field?

Suppose the client sends:

```
{
  "gender": "female",
  "tenure": 12,
  "monthlycharges": 50.0,
  "unexpected_field": 123
}
```

By default, Pydantic may ignore extra fields, depending on its configuration.

The workshop shows how to reject them explicitly:

```
from pydantic import BaseModel, ConfigDictclass Customer(BaseModel):    model_config = ConfigDict(extra="forbid")    gender: Literal["male", "female"]    tenure: int = Field(ge=0)    monthlycharges: float = Field(ge=0)
```

Now the additional field causes validation to fail.

This is helpful because it makes the API contract explicit.

## 8.6 What error does FastAPI return?

When request validation fails, FastAPI normally returns HTTP status code `422` with details about the invalid fields.

For example, the response may identify that:

- A required field is missing.
- A categorical value is not permitted.
- A numerical field violates its constraints.
- An unexpected field was supplied.

This is better than allowing invalid input to travel into the model and fail later in an unpredictable way.

Remember: Pydantic validates the structure and rules you've declared. It doesn't automatically know whether the customer information is factually true or whether the model itself is accurate.

# Part 9 — Environment management with `uv`

So far, we've discussed the model and API. Now we need to make sure the software environment is reproducible.

## 9.1 What is an environment problem?

Imagine that Project A needs one version of scikit-learn, while Project B needs another version.

If both projects use the same global Python environment, installing or upgrading packages for one project might break the other.

An isolated environment helps prevent this.

The workshop uses `uv` for environment and dependency management, replacing the older Pipenv workflow.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 9.2 What is uv?

`uv` is a Python tool that manages:

- Project dependencies.
- Virtual environments.
- Python version configuration.
- Dependency lockfiles.
- Running commands within the project environment.

It helps you reproduce the same dependency setup across machines.

## 9.3 Initialize a project

The workshop demonstrates:

```
uv init
```

This creates project configuration files, including:

- `pyproject.toml`
- `.python-version`

It may also create starter code and other files, depending on the version and initialization options.

### What is `pyproject.toml`?

This file describes project configuration and dependencies.

For example, it records which packages the project needs.

### What is `.python-version`?

This file specifies the Python version associated with the project.

### What is `.venv`?

This is the project's isolated virtual environment, created by the normal `uv` workflow.

It contains the packages used by that project rather than requiring you to install everything into global Python.

## 9.4 Add dependencies

The workshop uses commands like:

```
uv add scikit-learn fastapi uvicorn
```

This adds the required packages to the project and resolves their dependencies.

The command also creates or updates the project's environment and lockfile.

For testing the API from a separate Python script, the workshop adds `requests` as a development dependency:

```
uv add --dev requests
```

A development dependency is needed during development or testing but may not be needed by the running production API.

## 9.5 What is `uv.lock`?

The `uv.lock` file records the resolved dependency versions.

Why is this important?

Suppose your application uses:

- FastAPI.
- scikit-learn.
- NumPy.
- Uvicorn.

Each package may depend on other packages. The lockfile records the resolved set so the environment can be reproduced more reliably.

The difference is:

| File             | Main purpose                                     |
| ---------------- | ------------------------------------------------ |
| `pyproject.toml` | Describes project configuration and dependencies |
| `uv.lock`        | Records resolved dependency versions             |
| `.venv`          | Contains the installed project environment       |

## 9.6 Run commands using uv

Instead of calling Python or Uvicorn directly, the workshop uses `uv run`:

```
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

Or:

```
uv run python test.py
```

`uv run` executes the command within the project's managed environment.

## 9.7 Install dependencies for an existing project

Suppose you download a project that already contains its configuration and lockfile.

You can run:

```
uv sync
```

This synchronizes the project environment with its declared dependencies.

In situations where you must preserve the lockfile exactly, the stricter form is:

```
uv sync --locked
```

The `--locked` option fails if the lockfile would need to change.

### Why does this matter for machine learning?

A saved model may depend on specific versions of Python and machine learning libraries.

Using a controlled environment reduces compatibility problems and makes it easier to reproduce predictions.

# Part 10 — Docker and containerization

This section takes us from an application that runs on your computer to an application packaged with its environment.

## 10.1 What problem does Docker solve?

Suppose your API works on your computer because you have installed the right Python version, libraries, and files.

You send the code to another computer, but the application fails because the environment is different.

Docker helps package the application and its runtime dependencies into a reproducible image.

Three terms are essential.

Dockerfile

The instructions for building the image.

Docker image

The packaged template containing the application and its required environment.

Container

A running instance of an image.

A simple analogy:

- The Dockerfile is the recipe.
- The image is the prepared package.
- The container is the package running.

## 10.2 Understand the Dockerfile

The updated workshop builds a Docker image for the prediction API using Python, `uv`, the dependency lockfile, the model, and Uvicorn.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



Here is a simplified version of the workshop's Dockerfile structure:

```
FROM python:3.13.5-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --locked

COPY predict.py model.bin ./

EXPOSE 9696

ENTRYPOINT [
    "uvicorn",
    "predict:app",
    "--host", "0.0.0.0",
    "--port", "9696"
]
```

This is an instructional representation of the workshop pattern. The exact Python image and dependencies should match the environment you're using.

Let's go through it line by line.

### `FROM`

```
FROM python:3.13.5-slim-bookworm
```

Selects the base image.

The image provides Python and a minimal Debian Bookworm operating system environment.

### `COPY --from`

```
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
```

Copies the `uv` and `uvx` executables from another image into the image being built.

This is an example of copying selected files from another image rather than installing the tool from scratch.

For production builds, pinning the tool image version or digest improves reproducibility.

### `WORKDIR`

```
WORKDIR /code
```

Sets the working directory inside the container.

Subsequent relative file paths and commands use this directory.

### `ENV`

```
ENV PATH="/code/.venv/bin:$PATH"
```

Adds the project's virtual-environment executable directory to the command search path.

This allows the container to find tools installed in that environment.

### `COPY`

```
COPY pyproject.toml uv.lock .python-version ./
```

Copies the project configuration files into the image.

The Docker build then installs the dependencies:

```
RUN uv sync --locked
```

The lockfile is used to keep dependency resolution consistent.

Notice the order: the dependency files are copied before the application code. Docker can reuse the dependency installation layer when only application code changes.

### Copy the application and model

```
COPY predict.py model.bin ./
```

Copies the API code and saved model into the container.

If a required file isn't copied into the image, the application may fail to start.

### `EXPOSE`

```
EXPOSE 9696
```

Documents the port the application expects to use.

This instruction alone does not publish the port to your computer. Port publishing is handled when you run the container.

### `ENTRYPOINT`

```
ENTRYPOINT [
    "uvicorn",
    "predict:app",
    "--host", "0.0.0.0",
    "--port", "9696"
]
```

Specifies the command that starts the API when the container runs.

The container starts Uvicorn, which loads the FastAPI application from `predict.py`.

## 10.3 Build the Docker image

From the directory containing the Dockerfile, run:

```
docker build -t predict-churn .
```

Let's understand the command:

- `docker build` builds an image.
- `-t predict-churn` assigns it a name.
- `.` specifies the current directory as the build context.

The build context is the set of files Docker can access for instructions such as `COPY`.

That is why you should run the build from the correct directory.

## 10.4 Run the container

After building the image:

```
docker run -it --rm -p 9696:9696 predict-churn
```

The key part is:

```
-p 9696:9696
```

It means:

```
HOST PORT : CONTAINER PORT
    9696  :      9696
```

The host port is the port on your computer. The container port is the port where Uvicorn listens inside the container.

The other flags mean:

- `-i`: keep standard input open.
- `-t`: allocate a terminal.
- `--rm`: remove the container automatically when it stops.

`--rm` removes the stopped container, not the image.

## 10.5 Test the containerized application

With the container running, open:

- [http://127.0.0.1:9696/docs](http://127.0.0.1:9696/docs?utm_source=chatgpt.com)
- [http://127.0.0.1:9696/ping](http://127.0.0.1:9696/ping?utm_source=chatgpt.com), if that route exists in the application.

Then send a prediction request to `/predict`.

At this point, the API is running inside Docker, but you're still accessing it through your own computer.

The next step is to run the container on a cloud platform.

# Part 11 — Deploying the model to the cloud

## 11.1 What is cloud deployment?

So far, the service has been running on your own computer.

That means it normally isn't available to other people when your computer is shut down or disconnected.

Cloud deployment runs the application on infrastructure hosted by a cloud provider.

The updated workshop demonstrates Fly.io and also mentions alternatives such as Google Cloud Run and AWS App Runner. The older module material also covers AWS Elastic Beanstalk.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



The general process is:

1. Package the application.
2. Deploy it to the cloud.
3. Obtain a reachable service URL.
4. Send requests to that URL.
5. Check that the service works remotely.

## 11.2 Why Docker helps with cloud deployment

Without containerization, you may need to configure the Python environment and dependencies separately on the cloud machine.

With Docker, you can provide the platform with a packaged application image.

The cloud platform then runs the container.

This helps make the deployed environment closer to the one you tested locally.

## 11.3 Fly.io deployment workflow

The workshop illustrates this flow:

```
fly auth signup
fly launch --generate-name
fly deploy
```

Here's what each step does.

### `fly auth signup`

Creates or authenticates your Fly.io account, as appropriate to the CLI workflow.

### `fly launch --generate-name`

Starts the process of creating an application and its configuration.

### `fly deploy`

Deploys the application using the configured deployment setup.

The exact prompts and commands may change as the platform evolves. On Windows, follow the current Fly.io installation instructions for your shell rather than assuming that a Unix installation command will work in PowerShell.

## 11.4 Test the deployed API

After deployment, the platform provides a reachable URL.

Instead of testing:

```
http://127.0.0.1:9696/predict
```

you send requests to the deployed service's URL.

The same client logic can be reused by changing the URL:

```
import requestsurl = "https://your-deployed-service.example/predict"response = requests.post(url, json=customer)print(response.status_code)print(response.json())
```

The example URL is a placeholder, not an actual deployed service.

If the remote request succeeds, it demonstrates that the application can serve predictions outside your local environment.

## 11.5 Remember to check pricing and clean up

Cloud resources may incur charges depending on the provider, service, configuration, and usage.

The workshop demonstrates destroying an application when you no longer need it:

```
fly apps destroy <app-name>
```

This is destructive: it removes the specified application. Use the correct application name and confirm that you really want to delete it.

The lesson here is that deployment involves both making a service available and managing the infrastructure it runs on.

# Part 12 — Put the entire lesson together

Let's review the complete workflow in one example.

Imagine that you are building a customer-churn prediction service.

Step 1: Train

Fit the vectorizer and classifier using historical data.

Step 2: Save

Store the fitted pipeline in a file using pickle.

Step 3: Separate the programs

Use `train.py` to train and save, and `predict.py` to load and serve.

Step 4: Build the API

FastAPI receives customer data, validates it, and calls the pipeline.

Step 5: Test

Use `/docs`, `curl`, or `requests` to check the responses.

Step 6: Package

Use `uv` to manage dependencies and Docker to package the application.

Step 7: Deploy

Run the container on a cloud platform and test the remote endpoint.

## What each tool is responsible for

| Tool                             | Responsibility                               |
| -------------------------------- | -------------------------------------------- |
| scikit-learn                     | Training and prediction                      |
| `Pipeline`                       | Combines preprocessing and the model         |
| `pickle`                         | Saves and restores the fitted pipeline       |
| FastAPI                          | Defines the HTTP API                         |
| Uvicorn                          | Runs the API server                          |
| Pydantic                         | Validates input and output data              |
| `requests`                       | Lets a Python program call the API           |
| `uv`                             | Manages Python dependencies and environments |
| Docker                           | Packages the application and its runtime     |
| Fly.io or another cloud platform | Runs the deployed application                |

# Final revision: what you should understand before moving on

Use this as your learning checklist.

### Module 05 mastery checklist

### 0/17

Explain why a trained model needs to be saved.

Explain serialization and deserialization.

Explain why preprocessing should be saved with the model.

Create a scikit-learn Pipeline and predict for one dictionary of features.

Explain why training and inference are separate tasks.

Explain HTTP, JSON, GET, POST, and endpoints.

Create a FastAPI application and run it with Uvicorn.

Explain what the command uvicorn predict:app means.

Test an API through Swagger UI, curl, and requests.

Explain BaseModel, Literal, Field, and ConfigDict in Pydantic.

Explain why invalid input should be rejected.

Explain pyproject.toml, uv.lock, .venv, uv run, and uv sync.

Explain the difference between a Dockerfile, image, and container.

Explain FROM, COPY, WORKDIR, RUN, EXPOSE, and ENTRYPOINT.

Build an image and understand Docker port mapping.

Explain how a containerized API is deployed to the cloud.

Explain why a deployed endpoint should be tested and cloud resources cleaned up.

My recommendation is to make sure you can explain each topic in your own words before trying to memorize commands. In this module, the most important skill is understanding how the model, API, environment, container, and cloud service fit together.

We've now covered the topics in the official updated workshop. We have not started the homework; we can do that after you're comfortable with the lesson.