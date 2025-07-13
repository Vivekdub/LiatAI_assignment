def enhance_frame(frame):
    # 1. Denoise
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # 2. Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)

    # 3. Histogram equalization
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return frame
