import cv2

# Image/Video
img_file = 'carimage2.jpeg'

# Pre-trained Car Classifier
classifier_file = 'car_detector.xml'

# Create Car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)


def track_image(img_file: str):

    # Create opencv image
    img = cv2.imread(img_file)

    # Convert to grayscale (needed for haar cascade)
    black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect Cars
    cars = car_tracker.detectMultiScale(black_n_white)

    # Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image with the cars spotted
    cv2.imshow('Car and Pedestrian Detector', img)

    # Dont autoclose (wait here in the code and listen for a key press)
    cv2.waitKey()

def track_video(video_file: str):
    # Create opencv video
    video = cv2.VideoCapture(video_file)

    while True:
        # Read the current frame
        read_successful, frame = video.read()

        if read_successful:
            # Convert to grayscale (needed for haar cascade)
            black_n_white = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        # Detect Cars
        cars = car_tracker.detectMultiScale(black_n_white)

        # Draw rectangles around the cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the image with the cars spotted
        cv2.imshow('Car and Pedestrian Detector', frame)

        # Dont autoclose
        cv2.waitKey(1)

track_video('car_tracker1.mp4')

print("Code Completed")