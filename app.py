from fastapi import FastAPI
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the trained model
class GNNModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=1):
        super(GNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Initialize FastAPI
app = FastAPI()

# Load trained model
model = GNNModel()
model.load_state_dict(torch.load("gnn_model.pth", map_location="cpu"))
model.eval()

@app.get("/")
def home():
    return {"message": "AI-Powered Quantum Drug Discovery API is Running!"}

@app.post("/predict/")
async def predict(data: dict):
    input_data = torch.tensor(data["features"], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_data).item()
    return {"prediction": prediction}
