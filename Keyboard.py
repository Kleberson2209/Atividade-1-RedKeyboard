import cv2
import numpy as np
import sounddevice as sd

# Gerar som de uma frequência específica
def play_frequency(frequency, duration=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    sd.play(wave, samplerate=sample_rate)
    sd.wait()

# Mapeamento do teclado (notas musicais)
NOTES = {
    "C": 261.63,  # Dó
    "D": 293.66,  # Ré
    "E": 329.63,  # Mi
    "F": 349.23,  # Fá
    "G": 392.00,  # Sol
    "A": 440.00,  # Lá
    "B": 493.88   # Si
}

# Dividir em áreas
def get_note_from_position(x, frame_width, note_area):
    section_width = frame_width // note_area
    keys = list(NOTES.keys())
    index = x // section_width
    return keys[min(index, len(keys) - 1)]

# Divisórias do Teclado 
def draw_keyboard_regions(frame, frame_width, note_area):
    section_width = frame_width // note_area
    y_start = 20
    for i, note in enumerate(NOTES.keys()):
        x_start = i * section_width
        x_end = (i + 1) * section_width
        cv2.line(frame, (x_start, y_start), (x_start, frame.shape[0]), (255, 0, 0), 2)
        cv2.putText(frame, note, (x_start + 10, y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Linhas
def apply_edge_detection(frame, lines_strength):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 1, edges_colored, lines_strength / 100, 0)

# Luz
def adjust_brightness(frame, light_strength):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if light_strength == 0:
        hsv[:, :, 2] = 0
    elif light_strength == 100:
        hsv[:, :, 2] = 255
    else:
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (light_strength / 100), 0, 255)
        
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Blur
def apply_blur(frame, blur_strength):
    if blur_strength > 0:
        return cv2.GaussianBlur(frame, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)
    return frame

# Kernel
def apply_morphological_operations(mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded

cap = cv2.VideoCapture(0)
cv2.namedWindow('Filtros')
cv2.resizeWindow('Filtros', 640, 480)

# Trackbars atualizadas
cv2.createTrackbar("Notas", "Filtros", 7, 7, lambda x: None)  # Limite  das notas
cv2.createTrackbar("Detectar", "Filtros", 120, 200, lambda x: None)  # Sensibilidade do vermelho
cv2.createTrackbar("Linhas", "Filtros", 0, 100, lambda x: None)  # Linhas 
cv2.createTrackbar("Luz", "Filtros", 99, 100, lambda x: None)  # Ajuste de luz 
cv2.createTrackbar("Sustentar", "Filtros", 50, 100, lambda x: None)  # Sustain
cv2.createTrackbar("Blur", "Filtros", 0, 10, lambda x: None)  # Blur 
cv2.createTrackbar("Kernel", "Filtros", 1, 10, lambda x: None)  # Kernel 

last_note = None
last_time_played = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Valor Trackbars
    note_area = max(1, cv2.getTrackbarPos("Notas", "Filtros"))
    red_area = min(200, cv2.getTrackbarPos("Detectar", "Filtros"))
    lines_strength = cv2.getTrackbarPos("Linhas", "Filtros")
    light_strength = cv2.getTrackbarPos("Luz", "Filtros")
    sustain_strength = cv2.getTrackbarPos("Sustentar", "Filtros") / 100.0
    blur_strength = cv2.getTrackbarPos("Blur", "Filtros")
    kernel_size = max(1, cv2.getTrackbarPos("Kernel", "Filtros"))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Limitar o vermelho
    red_intensity = 120  
    lower_red_1 = np.array([0, red_area, red_intensity])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, red_area, red_intensity])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Operações morfológicas
    mask = apply_morphological_operations(mask, kernel_size)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_x = x + w // 2
            note = get_note_from_position(center_x, frame.shape[1], note_area)
            
            if note != last_note:
                play_frequency(NOTES[note], duration=sustain_strength)
                last_note = note
            
            cv2.putText(frame, f"Nota: {note}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame = apply_edge_detection(frame, lines_strength)
    frame = adjust_brightness(frame, light_strength)
    frame = apply_blur(frame, blur_strength)

    draw_keyboard_regions(frame, frame.shape[1], note_area)

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Filtros", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
