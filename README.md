# Hand-Sign-Recognition-using-OpenCV
Hand signs in a live video stream are converted to text.

The proposed approach uses a web camera to capture a picture/set of frames of a gesture and apply grayscale conversion to them. This enables the binary decision of finding the interested portion of the image.

After applying a filter on the grayscale image as a smoothing operation, thresholding is performed using a binarization method to separate the hand sign from the background.

Based on the width and height of the hand, the horizontal and vertical orientation of the hand is determined.

Follwing this, the bounding box and finger peaks in the image are determined.
Based on the finger peak positions and orientation of the hand, a  5-bit representation of the hand sign is concluded. This 5 bit value, enables a comparison of signs with the corresponding letter/number notations. The sign detected is then converted to text that can be understood by the common public.

