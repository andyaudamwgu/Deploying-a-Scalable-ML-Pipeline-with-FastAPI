import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import process_data, apply_label
from ml.model import load_model

# Load model and encoder
model_path = "model/model.pkl"
encoder_path = "model/encoder.pkl"
model = load_model(model_path)
encoder = load_model(encoder_path)

# Define input data structure
class Data(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# Create FastAPI instance
app = FastAPI()


@app.get("/")
async def get_root():
    """Return a welcome message."""
    return {"message": "Hello from the Income Prediction API!"}


@app.post("/predict/")
async def post_inference(data: Data):
    """Make a model inference based on input data."""
    # Convert Pydantic model to dict
    data_dict = data.dict()
    # Convert to DataFrame with hyphenated column names
    data_df = pd.DataFrame([{k.replace("_", "-"): v for k, v in data_dict.items()}])
    
    # Process the data
    X, _, _, _ = process_data(
        data_df,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        label=None,  # No label for inference
        training=False,
        encoder=encoder,
        lb=None  # Not needed for inference
    )
    
    # Run inference
    pred = model.predict(X)
    result = apply_label(pred)
    return {"result": result}