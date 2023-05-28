from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit for uploaded files


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['file']

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        file_extension = f".{filename.split('.')[1]}"

        # Process the image and perform object detection
        # Load the class names
        class_names = []
        with open("yolo_files/coco.names", "r") as f:
            class_names = [line.strip() for line in f.readlines()]

        # Load the model and set up the network
        net = cv2.dnn.readNet("yolo_files/yolov3.weights", "yolo_files/yolov3.cfg")

        def process_objects(frame):
            # Preprocess the frame
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

            # Perform object detection
            net.setInput(blob)
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            outputs = net.forward(output_layers)

            # Post-process the detections
            conf_threshold = 0.5
            nms_threshold = 0.4
            class_ids = []
            boxes = []
            confidences = []

            (H, W) = frame.shape[:2]

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > conf_threshold:
                        center_x = int(detection[0] * W)
                        center_y = int(detection[1] * H)
                        width = int(detection[2] * W)
                        height = int(detection[3] * H)
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)

                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, width, height])

            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            # Visualize the results
            colors = np.random.uniform(0, 255, size=(len(class_ids), 3))

            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    label = class_names[class_id]
                    confidence = confidences[i]
                    color = colors[i]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 2)

        # Function to perform object detection on an image
        def detect_objects_image():
            frame = cv2.imread(file_path)
            process_objects(frame)
            retval, buffer = cv2.imencode(file_extension, frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64

        # Function to perform object detection on a video
        def detect_objects_video():
            # Open the video file
            video = cv2.VideoCapture(file_path)

            # Get video properties
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)

            # Define output codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output/output.mp4', fourcc, fps, (width, height, ))

            frame_count = 0

            while True:
                ret, frame = video.read()

                if not ret:
                    break

                # to speed up processing
                frame_count += 1
                if frame_count % 3 == 0:
                    continue

                process_objects(frame)
                out.write(frame)
                cv2.imshow("Output", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video.release()
            out.release()
            cv2.destroyAllWindows()

        # Process the input (image or video)
        if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Process input as image
            finalized_image = detect_objects_image()
            # Render the output template with the detected objects
            return render_template('image-output.html', file_extension=file_extension,
                                   finalized_image=finalized_image)

        elif file_path.endswith(('.mp4', '.avi', '.mkv')):
            # Process input as video
            detect_objects_video()
            return render_template('video-output.html')

    else:
        return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)
