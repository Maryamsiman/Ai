import os

from google import genai
from google.genai import types
from PIL import Image
import supervision as sv


client = genai.Client(api_key="AIzaSyDMTUgfcm6gQv9sBOrRHKDV4Ep7cWE81xk")

MODEL_NAME = "gemini-3-flash-preview"
IMAGE_PATH ="captured_image.png"
PROMPT = "Detect the object. Return JSON list with 'box_2d' and 'label'."
image = Image.open(IMAGE_PATH)
response = client.models.generate_content(
    model=MODEL_NAME,
    contents=[PROMPT, image],
    config=types.GenerateContentConfig(
        temperature=0.3
    )
)
detections = sv.Detections.from_vlm(
    vlm=sv.VLM.GOOGLE_GEMINI_2_5,
    result=response.text,
    resolution_wh=image.size
)
box_annotator = sv.BoxAnnotator(thickness=8)
label_annotator = sv.LabelAnnotator(text_scale=1.5)

annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated = label_annotator.annotate(scene=annotated, detections=detections)

sv.plot_image(annotated)