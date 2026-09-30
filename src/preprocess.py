import glob
import os

import cv2


INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
IMAGE_SIZE = (224, 224)


def remove_hair_artifacts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    return inpainted


def process_dataset():
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return

    for class_name in os.listdir(INPUT_DIR):
        class_input_path = os.path.join(INPUT_DIR, class_name)
        class_output_path = os.path.join(OUTPUT_DIR, class_name)

        if not os.path.isdir(class_input_path):
            continue

        os.makedirs(class_output_path, exist_ok=True)

        for image_path in glob.glob(os.path.join(class_input_path, "*.*")):
            if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image = cv2.imread(image_path)
            if image is None:
                continue

            cleaned = remove_hair_artifacts(image)
            resized = cv2.resize(cleaned, IMAGE_SIZE)
            save_path = os.path.join(class_output_path, os.path.basename(image_path))
            cv2.imwrite(save_path, resized)
            print(f"Processed: {save_path}")


if __name__ == "__main__":
    process_dataset()
