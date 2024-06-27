from solve import solve
import cv2

PATH = "test_images"
PHOTO_COUNT = 5


def capture_images(path):

    cap = cv2.VideoCapture(0)

    for i in range(PHOTO_COUNT):
        SIZE = 480
        while True:
            _, frame = cap.read()
            cv2.flip(frame, 1, frame)
            frame = frame[0:480, 80:560]
            cpy = frame.copy()
            pt_1 = (int(16 * SIZE / 64), int(38 * SIZE / 64))
            pt_2 = (int(48 * SIZE / 64), int(52 * SIZE / 64))
            # draw ellipse to put face in
            cv2.ellipse(cpy, (SIZE // 2, int(SIZE // 2)), (int(SIZE // 3.3), int(SIZE // 2.55)), 0, 0, 360, (0, 255, 0), 2)

            cv2.imshow("capture", cpy)

            if cv2.waitKey(1) & 0xFF == ord(" "):
                cv2.imwrite(f"{path}/{path}/image-ariel{i}.png", frame)
                break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    capture_images(PATH)
    solve(PATH)
