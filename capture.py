import cv2
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
print("Press SPACE to capture an image...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    if key % 256 == 32:  # SPACE pressed
        img_name = "captured_image.png"
        cv2.imwrite(img_name, frame)
        print(f"Image captured and saved as {img_name}")
        break
cap.release()
cv2.destroyAllWindows()