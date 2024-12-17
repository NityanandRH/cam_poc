from ultralytics import YOLO
import cv2
import numpy as np


# Load the best model
model = YOLO('runs/detect/helmet_detection2/weights/best.pt')  # Replace with the path to your best model

print(model.names)

# Load the image
image_path = 'test3.jpg'
image = cv2.imread(image_path)

# Run prediction
results = model(image_path, conf=0.1)

def calculate_center(box):
    """
    Calculate the center point of a bounding box.
    box format: [x_min, y_min, x_max, y_max]
    """
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def extract_rider_with_triple(image, rider_boxes):
    """
    Extract the regions corresponding to rider bounding boxes with triple riders.

    Args:
        image (np.array): The input image.
        rider_boxes (list): List of dictionaries containing rider boxes and head counts.

    Returns:
        List of cropped images of riders with triple riders.
    """
    cropped_riders = []  # List to store cropped images of riders with triple riders

    # Loop through each rider box
    for i, rider in enumerate(rider_boxes):
        # Check if the rider has 3 or more heads (indicating triple riders)
        if rider['heads'] >= 3:
            # Extract the bounding box coordinates for the rider
            rider_box = rider['box']
            x_min, y_min, x_max, y_max = rider_box

            # Crop the image using the rider bounding box
            cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]

            # Add the cropped image to the list
            cropped_riders.append(cropped_image)

    return cropped_riders


def assign_heads_to_riders_and_extract(image, results):
    """
    Assign heads to riders and extract rider images that contain triple riders.

    Args:
        image (np.array): The input image.
        results: The result from YOLO containing bounding boxes, class, and confidence.

    Returns:
        List of cropped images of riders with triple riders.
    """
    rider_boxes = []  # List of rider bounding boxes
    head_boxes = []  # List of head bounding boxes

    # Step 1: Parse YOLO results and separate rider and head boxes
    for result in results[0].boxes:
        box = result.xyxy.squeeze()  # Bounding box [x_min, y_min, x_max, y_max]
        confidence = result.conf[0]  # Confidence score
        cls = int(result.cls[0])  # Class index

        if cls == 2:  # Class 2 corresponds to the rider
            rider_boxes.append({'box': box, 'heads': 0})  # Add rider box with head count = 0
        elif cls == 0 or cls == 1:  # Helmet or Without Helmet class (head boxes)
            head_boxes.append(box)

    # Step 2: Assign heads to riders
    for head_box in head_boxes:
        head_center = calculate_center(head_box)
        min_distance = float('inf')
        nearest_rider = None

        # Find the nearest rider
        for rider in rider_boxes:
            rider_center = calculate_center(rider['box'])
            distance = np.linalg.norm(np.array(head_center) - np.array(rider_center))

            if distance < min_distance:
                min_distance = distance
                nearest_rider = rider

        # Assign the head to the nearest rider
        if nearest_rider:
            nearest_rider['heads'] += 1

    # Step 3: Extract rider regions with triple riders
    return extract_rider_with_triple(image, rider_boxes)


# Assign heads to riders and extract cropped regions with triple riders
cropped_riders = assign_heads_to_riders_and_extract(image, results)

# Save or display the cropped rider images containing triple riders
if cropped_riders:
    for idx, rider_image in enumerate(cropped_riders):
        cv2.imwrite(f"rider_triple_{idx + 1}.jpg", rider_image)  # Save the cropped image
        print(f"Triple rider image {idx + 1} saved.")
else:
    print("No triple riders detected.")



# Annotate and display the image
annotated_image = results[0].plot()
cv2.imshow('YOLOv8 Prediction', annotated_image)
cv2.waitKey(0)

