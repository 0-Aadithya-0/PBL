import PIL.Image
import cv2
from gtts import gTTS
import playsound
from google import genai

client = genai.Client(api_key="AIzaSyC0FzAL0ObZxuiHvVeF4-xKZhuyGLgGBXo")


def Frame_Getter():
    cam = cv2.VideoCapture(0)

    while True:

        ret, frame = cam.read( )
        frame = cv2.flip(frame, 1)
        if not ret:
            print("failed to grab frame")
            break

        cv2.imshow("Press Space To Capture", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE

            cv2.imwrite("image.jpg", frame)
            print("Image Captured Successfully")
            break

    cam.release( )

    cv2.destroyAllWindows( )


if __name__ == "__main__":
    Frame_Getter( )
    image = PIL.Image.open('image.jpg')
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=["give a discription of the what you see , You are a person trying to discribe you vision to a blind person(do not add any intial dialogs like : here is what I see , okay imagine this, The image shows)", image])

    print(response.text)
    tts = gTTS(text=response.text, lang='en')
    tts.save("image.mp3")
    playsound.playsound("image.mp3")
