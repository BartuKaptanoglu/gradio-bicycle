import cv2
import supervision as sv
from inference.models.utils import get_roboflow_model
import gradio as gr
from PIL import Image
import requests
model = get_roboflow_model(model_id="object-xos6g/2", api_key="#####")
from sahi.utils.file import download_from_url
url="https://raw.githubusercontent.com/BartuKaptanoglu/gradio-bicycle/main/bisiklet.jpg",
download_from_url(
    "https://raw.githubusercontent.com/BartuKaptanoglu/gradio-bicycle/main/bisiklet.jpg",
    "bisiklet1.jpg"
)
def func(image_path):
    image = cv2.imread(image_path)
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    im_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im_rgb)
    return gr.Plot(im)
example=[
    ["bisiklet1.jpg"]
]

demo = gr.Interface(func, gr.Image(type="filepath"), outputs="image",title="Yolo bike",examples=example)
demo.launch(debug=True)