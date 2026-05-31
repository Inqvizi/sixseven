import cv2
import mediapipe as mp
import math
import pygame

pygame.mixer.init()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

video_path = 'gato.mp4'
audio_path = 'gato.mp3'
cat_video_cap = None
webcam_cap = cv2.VideoCapture(0)
is_video_playing = False

prev_y1, prev_y2 = None, None
frames_without_movement = 0
MAX_TOLERANCE_FRAMES = 15
MOVEMENT_THRESHOLD = 0.05


def is_open_palm(lm):
    tips_and_bases = [(8, 6), (12, 10), (16, 14), (20, 18)]
    up_fingers = sum(1 for tip, base in tips_and_bases if lm[tip].y < lm[base].y)
    return up_fingers >= 3


while True:
    ret, frame = webcam_cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    open_palms_y = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark

            if is_open_palm(lm):
                open_palms_y.append(lm[9].y)

    is_moving = False

    if len(open_palms_y) == 2:
        y1, y2 = open_palms_y[0], open_palms_y[1]

        if prev_y1 is not None and prev_y2 is not None:
            dy1 = abs(y1 - prev_y1)
            dy2 = abs(y2 - prev_y2)

            if dy1 > MOVEMENT_THRESHOLD or dy2 > MOVEMENT_THRESHOLD:
                is_moving = True

        prev_y1, prev_y2 = y1, y2
    else:
        prev_y1, prev_y2 = None, None

    if is_moving:
        frames_without_movement = 0
        if not is_video_playing:
            cat_video_cap = cv2.VideoCapture(video_path)
            if cat_video_cap.isOpened():
                is_video_playing = True

                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play(-1)

                print("Тренд Six Seven! Відео та звук увімкнено!")
    else:
        if is_video_playing:
            frames_without_movement += 1
            if frames_without_movement > MAX_TOLERANCE_FRAMES:
                is_video_playing = False

                pygame.mixer.music.stop()

                if cat_video_cap:
                    cat_video_cap.release()
                cv2.destroyWindow("Sixseven")
                print("Рух зупинився: Відео вимкнено")

    if is_video_playing and cat_video_cap is not None:
        ret_vid, cat_frame = cat_video_cap.read()
        if not ret_vid:
            cat_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_vid, cat_frame = cat_video_cap.read()

        if ret_vid:
            cat_frame = cv2.resize(cat_frame, (400, 400))
            cv2.imshow("Sixseven", cat_frame)

    cv2.imshow("Main Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

webcam_cap.release()
if cat_video_cap: cat_video_cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()