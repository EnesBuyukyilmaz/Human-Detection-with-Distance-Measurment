import cv2
import darknet
import numpy as np
import Jetson.GPIO as GPIO
import time
from RPLCD.i2c import CharLCD
LED = 22
TRIG = 23
ECHO = 24

GPIO.setmode(GPIO.BOARD	)
GPIO.setup(LED,GPIO.OUT)
GPIO.output(LED,GPIO.LOW)
lcd = CharLCD(
    i2c_expander='PCF8574',
    address=0x27,
    port=1,
    cols=16,
    rows=2
)
lcd.cursor_pos = (0,0)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
def distance_cm():
    # Trigger pulse
    GPIO.output(TRIG, GPIO.HIGH)
    time.sleep(0.00001)  # 10 µs
    GPIO.output(TRIG, GPIO.LOW)

    # Echo timing
    timeout = time.time() + 0.04
    while GPIO.input(ECHO) == 0:
        if time.time() > timeout:
            return None
    start = time.time()

    while GPIO.input(ECHO) == 1:
        if time.time() > timeout:
            return None
    end = time.time()

    duration = end - start
    return (duration * 34300) / 2  # cm

GPIO.output(TRIG, GPIO.LOW)
# YOLO config
configPath = "cfg/yolov3.cfg"
weightPath = "yolov3.weights"
dataPath = "cfg/coco.data"

# Network yükle
network, class_names, class_colors = darknet.load_network(
    configPath,
    dataPath,
    weightPath,
    batch_size=1
)

# Kamera aç
cap = cv2.VideoCapture(0)   # /dev/video0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = darknet.network_width(network)
height = darknet.network_height(network)
poepleCount = 0

# Darknet image'i BİR KERE oluştur (performans için)
img_darknet = darknet.make_image(width, height, 3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Darknet için BGR → RGB resize
    frame_resized = cv2.resize(frame, (width, height))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Darknet'e uygun hale getir
    darknet.copy_image_from_bytes(img_darknet, frame_rgb.tobytes())

    # Tespit yap
    detections = darknet.detect_image(network, class_names, img_darknet)
    personCount = 0
    # Sonuçları çiz (SADECE PERSON)
    for label, confidence, bbox in detections:
        if label != "person":
            continue
        personCount +=1
        x, y, w, h = bbox
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_resized,
            f"person {confidence}%",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imshow("YOLOv3 Jetson Nano", frame_resized)
    if True:
        lcd.clear()
        d = distance_cm()
        if d is None:
            lcd.cursor_pos = (1,0)
            lcd.write_string("Kimse bulunumadi")
            print("")

        else:
            
            lcd.cursor_pos = (1,0)
            lcd.write_string(f"Mesafe: {d:.1f} cm")
        if d is not None and personCount>0 and d <=100 :
            GPIO.output(LED,GPIO.HIGH)
        else:
            GPIO.output(LED,GPIO.LOW)
    print( "person count: ", personCount)
    lcd.cursor_pos = (0,0)
    string = "Insan Sayisi: " + str(personCount)
    lcd.write_string(string)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()



