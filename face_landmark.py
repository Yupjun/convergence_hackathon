def face_landmark(image_name):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        image = cv2.imread(image_name)
        faces = detector(image)
        landmarks_list = []
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            landmarks = predictor(image, face)
            location = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                location.append((x,y))
            landmarks_list.append(location)

        return landmarks_list[0]