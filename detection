from content.maskrcnn.m_rcnn import *
from content.maskrcnn.visualize import get_mask_contours, draw_mask

def detect_objects_in_images(image_folder, model_path):
    test_model, inference_config, class_names = load_inference_model(3, model_path)

    # Set mask colors for each class
    pistol_color = (255, 0, 0)  # Red for pistol
    rifle_color = (0, 0, 255)  # Blue for rifle

    # Iterate over images in the folder
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            #image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(image)

    # Detect results for each image
    annotated_images = []
    max_text_width = 0
    max_text_height = 0
    for image in images:
        r = test_model.detect([image])[0]

        object_count = len(r["class_ids"])
        has_pistol = False
        has_rifle = False
        for i in range(object_count):
            # 1. Mask
            mask = r["masks"][:, :, i]
            contours = get_mask_contours(mask)

            # 2. Class name
            class_name = class_names[r["class_ids"][i]]
            print(f"Object {i}: {class_name}")

            # 3. Bounding box
            y1, x1, y2, x2 = r['rois'][i]
            if class_name == 'pistol':
                has_pistol = True
                mask_color = pistol_color
            elif class_name == 'rifle':
                has_rifle = True
                mask_color = rifle_color
            else:
                mask_color = (0, 0, 0)  # Set black color for other classes
            # cv2.rectangle(img, (x1, y1), (x2, y2), mask_color, thickness=2)

            # 4. Draw mask with corresponding color for each class
            for cnt in contours:
                if class_name == 'pistol':
                    image = draw_mask(image, [cnt], pistol_color)
                elif class_name == 'rifle':
                    image = draw_mask(image, [cnt], rifle_color)

        # 5. Add text to image
        text = ""
        color = (0, 0, 0)
        if has_pistol and has_rifle:
            text = "Both"
        elif has_pistol:
            text = "pistol"
            color = pistol_color
        elif has_fracture:
            text = "rifle"
            color = rifle_color
        else:
            text = "SAFE"  # No classes found
            color = (0, 255, 0)  # Green color for "OK"

        if text:
            font_scale = 8  # Adjust the font scale as per your requirement
            font_thickness = 1  # Adjust the font thickness as per your requirement
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_width = text_size[0]
            text_height = text_size[1]
            if text_width > max_text_width:
                max_text_width = text_width
            if text_height > max_text_height:
                max_text_height = text_height

            # Store text and color information for later use
            image_info = {
                "image": image,
                "text": text,
                "color": color
            }
            annotated_images.append(image_info)

    # Adjust font scale based on the maximum text width and height
    max_font_scale = 0.8  # Adjust as per your requirement
    font_scale = min(max_font_scale, max_text_width / 300, max_text_height / 300)

    a =125

    # Display annotated images
    num_images = len(annotated_images)
    if num_images > 0:
        num_rows = (num_images + 3) // 4  # Calculate number of rows
        num_cols = min(num_images, 4)  # Maximum number of columns is 4

        # Create a blank canvas for arranging images
        canvas_height = num_rows * a
        canvas_width = num_cols * a
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Arrange images on the canvas
        for i, image_info in enumerate(annotated_images):
            row = i // 4
            col = i % 4
            x = col * a
            y = row * a
            image = image_info["image"]
            text = image_info["text"]
            color = image_info["color"]

            # Resize image
            image = cv2.resize(image, (a, a))

            # Add text to the image with adjusted font scale
            font_thickness = 1  # Adjust the font thickness as per your requirement
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_width = text_size[0]
            text_height = text_size[1]
            text_x = (a - text_width) // 2
            # text_y = (300 + text_height) // 2
            text_y = text_height + 10
            cv2.putText(image, text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

            # Add image to the canvas
            canvas[y:y + a, x:x + a] = image

        cv2.imshow("Output Images", canvas)
        cv2.waitKey()
    else:
        print("No images found in the specified folder.")

# Example usage
image_folder_path = "content/test_images"
model_file_path = "content/mask_rcnn_object_0030.h5"

detect_objects_in_images(image_folder_path, model_file_path)
