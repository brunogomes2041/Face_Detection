from deepface import DeepFace
import cv2
import os

def identificar_pessoa(frame, db_path):
    try:
        resultado = DeepFace.find(img_path=frame, db_path=db_path, model_name="VGG-Face")
        if resultado and not resultado[0].empty:
            nome = resultado[0].iloc[0]['identity']
            return os.path.basename(nome).split('.')[0]
        else:
            return "Nao identificado"
    except Exception as e:
        return f"Erro: {str(e)}"

caminho_banco = "database"
captura = cv2.VideoCapture(0)

if not os.path.exists(caminho_banco):
    print(f"O caminho do banco de dados '{caminho_banco}' nao foi encontrado.")
    exit()

print("Pressione 'q' para sair.")

while True:
    ret, frame = captura.read()
    if not ret:
        print("Erro ao acessar a câmera.")
        break

    temp_frame_path = "temp_frame.jpg"
    cv2.imwrite(temp_frame_path, frame)

    nome = identificar_pessoa(temp_frame_path, caminho_banco)

    if nome == "Nao identificado":
        aviso = "Rosto nao identificado"
    elif nome.startswith("Erro"):
        aviso = "Rosto nao encontrado"
    else:
        aviso = nome

    cv2.putText(frame, aviso, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Reconhecimento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
if os.path.exists(temp_frame_path):
    os.remove(temp_frame_path)
