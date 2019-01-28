import cv2
import utils as u

# Path to yolo configuration file
model_configuration = "../yolo_files/yolov3.cfg"

# Path to yolo pre-trained weights
model_weights = "../yolo_files/yolov3.weights"

# Path to text file containing class names
classes_file = "../yolo_files/coco.names"

def main():
    print("AAAAAHHHHH")

    # Read class names and generate different colors for different classes
    classes, colors = u.read_classes_and_generate_colors( classes_file )

    # Read pre-trained model and config file
    net = cv2.dnn.readNetFromDarknet( model_configuration, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Define a window to show the cam stream on it
    window_title= "YOLOv3"   
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    # --- Processing real time video ---S
    # Webcam input
    cap = cv2.VideoCapture(0)

    while cv2.waitKey(1) < 0:
        # get frame from the video
        hasFrame, frame = cap.read()

        blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (416,416), [0,0,0], True, crop=False)

        # Sets the input to the network
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward( u.get_output_layers(net) )
        # Remove the bounding boxes with low confidence
        frame = u.post_process(frame, outs, classes, colors)

        # Put efficiency information. The function getPerfProfile returns the 
        # overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.imshow(window_title, frame)

if __name__ == '__main__':
    main()