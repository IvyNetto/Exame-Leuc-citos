# import cv2
# import numpy as np

# # Função para detectar leucócitos
# def detectar_leucocitos(imagem_path):
#     # Carregar a imagem
#     imagem = cv2.imread(imagem_path)
#     if imagem is None:
#         print("Imagem não encontrada!")
#         return
    
#     # Converter para espaço de cor HSV
#     imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

#     # Definir intervalo de cor para leucócitos (geralmente na faixa de azul a roxo)
#     # Esses valores podem precisar de ajustes conforme a coloração da amostra
#     lower_blue = np.array([100, 50, 50])  # Limite inferior da cor
#     upper_blue = np.array([140, 255, 255])  # Limite superior da cor

#     # Criar uma máscara que detecta apenas os pixels dentro do intervalo de cor
#     mascara = cv2.inRange(imagem_hsv, lower_blue, upper_blue)

#     # Aplicar a máscara à imagem original
#     resultado = cv2.bitwise_and(imagem, imagem, mask=mascara)

#     # Encontrar contornos dos leucócitos
#     contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Desenhar contornos na imagem original
#     for contorno in contornos:
#         if cv2.contourArea(contorno) > 500:  # Ajuste o tamanho mínimo do contorno
#             (x, y, w, h) = cv2.boundingRect(contorno)
#             cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Exibir os resultados
#     cv2.imshow("Imagem Original com Leucocitos Detectados", imagem)
#     cv2.imshow("Mascara", mascara)
#     cv2.imshow("Resultado", resultado)

#     # Aguardar até que uma tecla seja pressionada e depois fechar as janelas
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Exemplo de uso
# detectar_leucocitos("dataset/treino/images/BloodImage_00343.jpg")


# ==================================================================================================================


import cv2
import numpy as np

# Função para detectar leucócitos e analisar granularidade
def detectar_leucocitos_com_granularidade(imagem_path):
    # Carregar a imagem
    imagem = cv2.imread(imagem_path)
    if imagem is None:
        print("Imagem não encontrada!")
        return
    
    # Converter para escala de cinza
    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro de suavização para reduzir o ruído
    imagem_blur = cv2.GaussianBlur(imagem_gray, (5, 5), 0)

    # Detecção de bordas com Sobel (para detectar detalhes da granularidade)
    grad_x = cv2.Sobel(imagem_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(imagem_blur, cv2.CV_64F, 0, 1, ksize=3)

    # Calcular a magnitude do gradiente
    magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalizar a magnitude para uma melhor visualização
    magnitude = np.uint8(cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX))

    # Definir um limiar para a granularidade
    limiar_granularidade = 40  # Ajuste esse valor conforme necessário

    # Detectar contornos dos leucócitos na imagem
    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mascara = cv2.inRange(imagem_hsv, lower_blue, upper_blue)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Analisar granularidade de cada leucócito detectado
    for contorno in contornos:
        if cv2.contourArea(contorno) > 500:  # Filtrar contornos pequenos
            (x, y, w, h) = cv2.boundingRect(contorno)
            leucocito = imagem_gray[y:y+h, x:x+w]  # Extrair a região do leucócito

            # Calcular a variação de intensidade na região do leucócito
            leucocito_grad = magnitude[y:y+h, x:x+w]
            media_grad = np.mean(leucocito_grad)

            # Classificar o leucócito como granulado ou não granulado
            if media_grad > limiar_granularidade:
                tipo = "Granular"
                cor = (0, 255, 0)  # Verde para granulado
            else:
                tipo = "Agranular"
                cor = (0, 0, 255)  # Vermelho para não granulado

            # Desenhar o contorno e o texto
            cv2.rectangle(imagem, (x, y), (x + w, y + h), cor, 2)
            cv2.putText(imagem, tipo, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

    # Exibir a imagem com os resultados
    cv2.imshow("Resultado", imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

a=1
i=0
while a!=0 or i<410:
    if a == 0 or i>410:
        break
    a=int(input("Continuar? Padrão => Sim\n>>") or "1")
    detectar_leucocitos_com_granularidade(f"dataset/treino/images/BloodImage_{i:05}.jpg")
    i+=1
print("Encerrado")

