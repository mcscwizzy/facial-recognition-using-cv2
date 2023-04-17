import cv2

# initiate webcam instance
webcam = cv2.VideoCapture(0)

# import trained face data
# this is predefined training data from cv2 on github
trained_face_data = cv2.CascadeClassifier("face_detection.xml")

# eternally loop until q is pressed
while True:

    # vidoes are collection of images so we need to read this webcam as it comes in
    succesful_webcam_read, frame = webcam.read()

    # convert video feed to grayscale as cv2 only reads grayscale
    grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get the face coordinates for the square we will draw
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_image)

    # draw the square on the greyscale image
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    # display the video feed in a window
    cv2.imshow("My Webcam", frame)
    key = cv2.waitKey(1)

    # wxit program if the q key is pressed
    if key == 1 or key == 113:
        break

# release the wevcam resource
webcam.release()
