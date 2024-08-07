import cv2
import supervision as sv
from inference.models.utils import get_roboflow_model
import gradio as gr
from PIL import Image

model = get_roboflow_model(model_id="object-xos6g/2", api_key="RZuthAj7pTSv5sx7l4sg")


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


demo = gr.Interface(func, gr.Image(type="filepath"), outputs="image")
demo.launch(debug=True)