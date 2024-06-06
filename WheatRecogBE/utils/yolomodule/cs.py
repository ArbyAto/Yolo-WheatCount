import torch
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords

# Set image size and model path
imgsz = [640, 640]
model_path = "runs/train/best.pt"

# Load the entire model
model = torch.load(model_path)
model.eval()  # Set model to evaluation mode

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Source image
source = "../wheat_yolo_format/images/train/00001.jpg"

# Load the image
dataset = LoadImages(source, img_size=imgsz)

# Inference settings
augment = False  # Augmented inference
visualize = False  # Visualize features

# Perform inference
for path, im, im0s, vid_cap, s in dataset:
    im = torch.from_numpy(im).to(device)
    im = im.float()  # Convert image to float
    im /= 255.0  # Normalize image to 0-1 range
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    # Inference
    pred = model(im, augment=augment, visualize=visualize)[0]

    # Apply Non-Max Suppression (NMS)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()

            # Print results
            for *xyxy, conf, cls in det:
                print(f"Class: {cls}, Confidence: {conf}, BBox: {xyxy}")

# Note: You can further add visualization or saving the detection results as needed.
