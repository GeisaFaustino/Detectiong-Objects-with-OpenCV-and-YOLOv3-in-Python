import os
import cv2
import numpy as np

# Function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, colors):
    
    label = '%.2f' % confidence
    color = colors[class_id]
         
    # Get the label for the class name and its confidence
    if classes:
        assert(class_id < len(classes))
        label = '%s:%s' % (classes[class_id], label)
        
    # Draw a bounding box.
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    
    #Display the label at the top of the bounding box    
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img


# Function to get output layers names in the architecture
def get_output_layers( net ):
    layer_names = net.getLayerNames( )
    output_layers = [ layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers() ]
    return output_layers


# Function remove the bounding boxes with low confidence using non-maxima suppression.
def post_process( frame, outputs, classes, colors, conf_threshold = 0.5, nms_threshold = 0.4):
    frame_height, frame_width = frame.shape[:2]
    
    class_ids = []
    confidences = []
    bounding_boxes = []
    
    # For each detetion from each output layer get the confidence, 
    # class id, bounding box params and ignore weak detections
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax( scores )
            conf = scores[ class_id ]
            if conf > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                x = center_x - w / 2
                y = center_y - h / 2

                class_ids.append( class_id )
                confidences.append( float(conf) )
                bounding_boxes.append( [x, y, w, h] )
    
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes( bounding_boxes, confidences, conf_threshold, nms_threshold )

    frame_out = frame.copy()
        
    for i in indices:
        i = i[0]
        x, y, w, h = bounding_boxes[i]
        frame_out = draw_bounding_box( frame_out, class_ids[i], confidences[i], 
                                       round(x), round(y), round(x+w), round(y+h),
                                       classes, colors )
    
    return frame_out   

# Read class names from text file and generate different colors for different classes
def read_classes_and_generate_colors( classes_file ):
    # Read class names from text file
    classes = None
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Generate different colors for different classes
    colors = np.random.uniform( 0, 255, size = ( len(classes), 3 ) )

    return classes, colors