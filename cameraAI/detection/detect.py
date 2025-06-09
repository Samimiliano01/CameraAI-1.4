import numpy as np

# 17 classes - make sure these are your actual class names exactly
LABELS = [
    "cans", "cardboard", "colored glass bottles", "face mask", "glass bottle",
    "HDPE", "LDPE", "PET", "PVC", "paper bag", "paper cup",
    "paperboard", "peel", "pile of leaves", "rags", "styrofoam", "tetra pak"
]

def postprocess_blob_output(raw_output, conf_threshold=0.6):
    """
    Postprocesses raw output from the blob without sigmoid.
    Assumes raw_output shape: (num_detections, 4 + num_classes)
    """
    if isinstance(raw_output, list):
        raw_output = np.array(raw_output)

    num_classes = len(LABELS)
    detections = raw_output.reshape(-1, 4 + num_classes)
    print(len(detections))
    results = []
    for det in detections:
        bbox = det[:4]  # bbox coords (x,y,w,h)
        class_scores = det[4:]  # raw class scores

        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]

        if class_conf < conf_threshold:
            continue
        else:
            pass
            # print(class_scores)
            # print(class_id)
            # print(class_conf)
        label = LABELS[class_id]
        results.append(label)

    print(len(results))
    return results
