import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import dlib
from matplotlib import pyplot as plt
import cv2

#Definindo função EAR que calcula o aspecto do olho (tamanho) em tempo real
def EAR(olho):
    A = dist.euclidean(olho[1],olho[5])
    B = dist.euclidean(olho[2],olho[4])
    C = dist.euclidean(olho[0],olho[3])
    ear = (A+B)/(2.0*C)
    return ear
#Criando as variáveis que utilizaremos no código:
#Contadores:
cont=0
quadros = 0
#Vetor que vai receber o EAR:
vet= []
#Variáveis que vão definir o limiar do tamanho do olho para parâmetro de aberto/fechado e o número de frames que vai ser
# utilizado como parâmetro para definir uma piscada:
OLHO_AR_THRESH = 0.1
OLHO_AR_FRAMES = 3
#Variável que vai receber o total de vezes que os olhos se fecharam:
total=0
#Iniciando a captura do vídeo utilizando OpenCV:
captura = cv2.VideoCapture("C:/Users/jl_sa/Downloads/11-0- BE Modesta Bento_pós_10_08_2021.MOV")
#Recebendo a quantidade de frames por segundo do vídeo:
fps= captura.get(cv2.CAP_PROP_FPS)
#Variavel que recebe a taxa de frames por segundo do vídeo em um vetor:
txdeframes=[]
#Variável que recebe o detector pré-treinado usando SVM+HOG da biblioteca Dlib
detector = dlib.get_frontal_face_detector()
#Variável que busca o arquivo contendo os pontos para realizar a predição dos pontos na face das pessoas identificadas:
predictor =  dlib.shape_predictor('eye_predictor.dat')
#Selecionando apenas os pontos referentes aos olhos no código utilizando a biblioteca IMUTILS para facilitar:
(lStart,lEnd) = (0,6)
(rStart,rEnd) = (6,12)
#Início da iteração para leitura do vídeo a partir da varíavel captura
#a variável frame recebe cada frame do vídeo enquanto existirem frames
#a variável ret funciona como uma flag para identificar se existe ou não mais frames
#tudo isso é feito utilizando OpenCV
while True:
    ret, frame = captura.read()
    #Variavel quadros recebe o número de frames que existem no vídeo
    quadros+=1
    #Condicional que define quando não existirem mais frames, deve-se sair do laço
    if not ret:
        break
    #Fazendo resize da imagem recebida usando IMUTILS para uma analise mais precisa da SVM+HOG
    frame = imutils.resize(frame, width=500)
    #Variável recebe o frame do vídeo em branco e preto utilizando a função cvtColor do OpenCV
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Variável faces recebe o detector, que recebe como parâmetro os frames já convertidos pra preto e branco
    faces = detector(gray_frame,0)
    #Dentro desse laço, contamos cada ponto que é identificado por faces que é o nosso detector de pontos para o frame
    #convertido em P&B
    for face in faces:
        #a variavel shape recebe a funçâo predictor, que realiza a predição dos pontos e recebe como parâmetros a imagem
        # e o detector de pontos
        shape = predictor(gray_frame,face)
        #a variável shape recebe a si mesma como uma matriz utilizando a função do IMUTILS face_utils.shape_to_np
        shape = face_utils.shape_to_np(shape)
        #Estabelecemos como parâmetros as variaveis Olhoesq e Olhodir que vão receber os pontos associados aos olhos
        # junto com a predição feita em shape
        Olhoesq= shape[lStart:lEnd]
        Olhodir= shape[rStart:rEnd]
        #Fazemos o cálculo do EAR (Eye Aspect Ratio) em tempo real de cada olho e recebemos nas variáveis EAResq e EARdir
        EAResq = EAR(Olhoesq)
        EARdir = EAR(Olhodir)
        #A variável ear recebe a média entre os valores obtidos por EAResq e EARdir
        ear = (EAResq + EARdir)/2.0
        #Atrelando os valores de cada ear de cada 3 frames para dentro do vetor vet
        vet.append(ear)
        #Identificando o formato dos olhos (conectando os pontos) que encontramos usando as variáveis Olhoesq e Olhodir
        # usando a função convexHull do OpenCV
        Hullesq = cv2.convexHull(Olhoesq)
        Hulldir = cv2.convexHull(Olhodir)
        #Desenhando o formato dos pontos obtidos nos frames usando a função drawContours do OpenCV
        cv2.drawContours(frame, [Hullesq], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [Hulldir], -1, (0, 0, 255), 1)
        #Estabelecendo padrões de análise de quando o olho está aberto ou fechado e contando o número de vezes que a
        # pessoa pisca:
        # para contar uma piscada, o olho deve ter um ear menor do que o que estabelecemos como limiar. Caso isso ocorra
        # o contador cont recebe +1
        if ear < OLHO_AR_THRESH:
            cont+=1
        # para que o contador não exploda a contagem, se o olho não sofrer alterações em seu ear por mais de 3 quadros
        # o contador zera e a variável total recebe +1.
        elif cont>= OLHO_AR_FRAMES:
            total+=1
            cont =0
        #Printando o número de piscadas total feito pela pessoa e a taxa de variação do aspecto do olho (ear) na tela:
        cv2.putText(frame, "Piscadas: {}".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "ear: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #Aqui começam as análises gráficas:
    # para cada análise dentro do laço while, o vetor txdeframes recebe o número de quadros analisados a cada iteração
    # divido por 30 para termos um parâmetro de comparação dado em segundos, já que a taxa de captura do OpenCV é de
    # 30 frames/s
    txdeframes.append(quadros/30)
    # Printando na tela em tempo real o vídeo
    cv2.imshow('Contador', frame)
    # Definindo um botão (no caso ESC) para que o vídeo seja reproduzido. No caso, o valor de waitkey é definido como
    # True, e caso waitkey receba 27 (ESC) o programa sai do laço while, permitindo uma parada a qualquer momento.
    # feito usando a função waitKey do OpenCV
    key = cv2.waitKey(1)
    if key == 27:
        break
# ajustando o tamanho dos vetores para serem iguais na hora de plotar os gráficos:
tam1 = len(vet)
tam2 = len(txdeframes)
dif=tam1-tam2
if (dif !=0):
    txdeframes= txdeframes[0:dif]
#Plotando o gráfico de análise do número de piscadas/tempo usando MatPlotLib:
plt.plot(txdeframes,vet,c='r',lw='1',marker='o',ms='1')
plt.suptitle('EAR x Tempo')
plt.ylabel('EAR')
plt.xlabel('Segundos')
#plt.yticks([0.4,0.38,0.19],['Aberto','Variante','Fechado'])
plt.axis([0,(quadros)/30,-0.2,0.5])
plt.grid(True)
plt.show()
cv2.destroyAllWindows()
