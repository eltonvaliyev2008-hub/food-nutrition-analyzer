from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torchvision.models import resnet101
from torchvision import transforms
from PIL import Image
import pandas as pd
import io

app = FastAPI(title="Calorie Lens")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = [
    "apple_pie","baby_back_ribs","baklava","beef_carpaccio","beef_tartare",
    "beet_salad","beignets","bibimbap","bread_pudding","breakfast_burrito",
    "bruschetta","caesar_salad","cannoli","caprese_salad","carrot_cake",
    "ceviche","cheesecake","cheese_plate","chicken_curry","chicken_quesadilla",
    "chicken_wings","chocolate_cake","chocolate_mousse","churros","clam_chowder",
    "club_sandwich","crab_cakes","creme_brulee","croque_madame","cup_cakes",
    "deviled_eggs","donuts","dumplings","edamame","eggs_benedict","escargots",
    "falafel","filet_mignon","fish_and_chips","foie_gras","french_fries",
    "french_onion_soup","french_toast","fried_calamari","fried_rice",
    "frozen_yogurt","garlic_bread","gnocchi","greek_salad",
    "grilled_cheese_sandwich","grilled_salmon","guacamole","gyoza","hamburger",
    "hot_and_sour_soup","hot_dog","huevos_rancheros","hummus","ice_cream",
    "lasagna","lobster_bisque","lobster_roll_sandwich","macaroni_and_cheese",
    "macarons","miso_soup","mussels","nachos","omelette","onion_rings",
    "oysters","pad_thai","paella","pancakes","panna_cotta","peking_duck",
    "pho","pizza","pork_chop","poutine","prime_rib","pulled_pork_sandwich",
    "ramen","ravioli","red_velvet_cake","risotto","samosa","sashimi",
    "scallops","seaweed_salad","shrimp_and_grits","spaghetti_bolognese",
    "spaghetti_carbonara","spring_rolls","steak","strawberry_shortcake",
    "sushi","tacos","takoyaki","tiramisu","tuna_tartare","waffles"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5459, 0.4445, 0.3444],
        std= [0.2625, 0.2659, 0.2706]
    )
])

model_path = hf_hub_download(
    repo_id="EltonValiyev11/calorie-lens-model",
    filename="resnet101_best_model.pth"
)

model = resnet101()
model.fc = nn.Linear(2048, 101)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

df = pd.read_csv("nutrition.csv")

WEIGHTS = [50, 70, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"weights": WEIGHTS}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file:    UploadFile = File(...),
    weight:  int        = Form(100)
):
    contents = await file.read()
    img      = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor   = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output  = model(tensor)
        _, pred = output.max(1)

    class_name = classes[pred.item()]

    row = df[
        (df["label"]  == class_name) &
        (df["weight"] == weight)
    ]

    if len(row) > 0:
        kalori   = row["calories"].values[0]
        protein  = row["protein"].values[0]
        karbohid = row["carbohydrates"].values[0]
        yag      = row["fats"].values[0]
        lif      = row["fiber"].values[0]
        seker    = row["sugars"].values[0]
        natrium  = row["sodium"].values[0]
    else:
        kalori = protein = karbohid = yag = lif = seker = natrium = "N/A"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "weights"   : WEIGHTS,
            "class_name": class_name,
            "weight"    : weight,
            "kalori"    : kalori,
            "protein"   : protein,
            "karbohid"  : karbohid,
            "yag"       : yag,
            "lif"       : lif,
            "seker"     : seker,
            "natrium"   : natrium,
        }
    )

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0"}
