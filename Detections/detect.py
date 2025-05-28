from ultralytics import YOLO

# model = YOLO("AImodel/datasets/solidwaste_project/yolov8n_v16_results/weights/best.pt")
def detect_litter(image, model, confidence:float = 0.6) -> list[str]:
    """
    Detecteert afval in een afbeelding en geeft een lijst met labels terug
    van objecten met confidence die groter is dan de meegegeven drempel.

    Parameters:
    -----------
    image : afbeelding
        Input afbeelding voor detectie.
    confidence : float, optioneel
        Minimum confidence threshold voor detecties (default is 0.6).

    Returns:
    --------
    list[str]
        Lijst met gedetecteerde labels boven de confidence drempel.
    """
    results = model(image)
    detected_types = [
        results[0].names[int(box.cls)]
        for box in results[0].boxes
        if float(box.conf) > confidence
    ]
    return detected_types

