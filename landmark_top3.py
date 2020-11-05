def landmark_top3(arr_2d,landmarks):
    mouth = []
    top3 = {}
    result = []
    for i in range(48,68):
        x = landmarks[i][0]
        y = landmarks[i][1]
        mouth.append(arr_2d[x][y])
    top3['mouth'] = sum(mouth) / len(mouth)

    right_eyebrow = []
    for i in range(17,22):
        x = landmarks[i][0]
        y = landmarks[i][0]
        right_eyebrow.append(arr_2d[x][y])
    top3['right_eyebrow'] = sum(right_eyebrow) / len(right_eyebrow)

    left_eyebrow = []
    for i in range(22,27):
        x = landmarks[i][0]
        y = landmarks[i][0]
        left_eyebrow.append(arr_2d[x][y])
    top3['left_eyebrow'] = sum(left_eyebrow) / len(left_eyebrow)

    right_eye = []
    for i in range(36,42):
        x = landmarks[i][0]
        y = landmarks[i][0]
        right_eye.append(arr_2d[x][y])
    top3['right_eye'] = sum(right_eye) / len(right_eye)

    left_eye = []
    for i in range(42,48):
        x = landmarks[i][0]
        y = landmarks[i][0]
        left_eye.append(arr_2d[x][y])
    top3['left_eye'] = sum(left_eye) / len(left_eye)

    nose = []
    for i in range(27,35):
        x = landmarks[i][0]
        y = landmarks[i][0]
        nose.append(arr_2d[x][y])
    top3['nose'] = sum(nose) / len(nose)   

    jaw = []
    for i in range(0,17):
        x = landmarks[i][0]
        y = landmarks[i][0]
        jaw.append(arr_2d[x][y])
    top3['jaw'] = sum(jaw) / len(jaw)
    sorted_top3 = {k:v for k,v in sorted(top3.items(), key=lambda item : item[1])}
    sorted_top3 = collections.OrderedDict(sorted_top3)

    result.append(sorted_top3.popitem())
    result.append(sorted_top3.popitem())
    result.append(sorted_top3.popitem())
    return result