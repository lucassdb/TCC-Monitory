from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
import pyrebase
from itertools import zip_longest

t0 = time.time()

def run():
	# construir o argumento
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=False,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	# confidence default 0.4
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())

	# inicializar a lista de rótulos de classe MobileNet SSD foi treinado para
	# detect
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	#carregar nosso modelo serializado do disco
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# se um caminho de vídeo não foi fornecido, pegue uma referência para a câmera ip
	if not args.get("input", False):
		print("[INFO] Starting the live stream..")
		vs = VideoStream(config.url).start()
		time.sleep(2.0)

	# caso contrário, pegue uma referência ao arquivo de vídeo
	else:
		print("[INFO] Starting the video..")
		vs = cv2.VideoCapture(args["input"])

	# inicializar o gravador de vídeo (vamos instanciar mais tarde, se necessário)
	writer = None

	# inicializar as dimensões do quadro (vamos defini-los assim que lermos
	# o primeiro quadro do vídeo)
	W = None
	H = None

	# instanciar nosso rastreador de centróide e inicializar uma lista para armazenar
	# cada um de nossos rastreadores de correlação dlib, seguido por um dicionário para
	# mapeia cada ID de objeto exclusivo para um TrackableObject
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}
	# credenciais do Firebase
	firebaseConfig = {
		"apiKey": "AIzaSyCCE4yveEuH_09vThP5U1J0Oi3d1-r2muY",
		"authDomain": "teste-90925.firebaseapp.com",
		"databaseURL": "https://teste-90925.firebaseio.com",
		"projectId": "teste-90925",
		"storageBucket": "teste-90925.appspot.com",
		"messagingSenderId": "267430548654",
		"appId": "1:267430548654:web:475cb40c90b836b32fc93f"
	}
 	#inicializar o firebase
	firebase = pyrebase.initialize_app(firebaseConfig)

	# inicializa o número total de frames processados ​​até agora, junto
	# com o número total de objetos que se moveram para cima ou para baixo
	totalFrames = 0
	totalDown = 0
	totalUp = 0
	x = []
	empty=[]
	empty1=[]
	dados = firebase.database() # variavel para receber as informações do firebase

	# iniciar o estimador de taxa de frames por segundo
	fps = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)

	# faz um loop sobre os frames do stream de vídeo
	while True:
		# pega o próximo quadro e controla se estivermos lendo de qualquer
		# VideoCapture ou VideoStream
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		# se estamos assistindo a um vídeo e não capturamos um quadro, então
		# chegou ao final do vídeo
		if args["input"] is not None and frame is None:
			break

		# redimensiona o quadro para ter uma largura máxima de 500 pixels (o
		# menos dados temos, mais rápido podemos processá-los) e, em seguida, converter
		# o quadro de BGR para RGB para dlib
		frame = imutils.resize(frame, width = 500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# se as dimensões do quadro estiverem vazias, defina-as
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# se deveríamos estar gravando um vídeo no disco, inicialize
		# the writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		# inicializar o status atual junto com nossa lista de
		# retângulos de caixa retornados por (1) nosso detector de objetos ou
		# (2) os rastreadores de correlação
		status = "Waiting"
		rects = []

		# verifique se devemos executar um computador
		# método de detecção de objeto para ajudar nosso rastreador
		if totalFrames % args["skip_frames"] == 0:
			# set the status and initialize our new set of object trackers
			status = "Detectando"
			trackers = []

			# converter o quadro em um blob
			# rede e obter as detecções
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# loop nas detecções
			for i in np.arange(0, detections.shape[2]):

				# extrair a confiança (ou seja, probabilidade) associada
				# com a previsão
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by requiring a minimum
				# confidence
				if confidence > args["confidence"]:
					# filtrar detecções fracas exigindo um mínimo
					# confiança
					idx = int(detections[0, 0, i, 1])

					# se o rótulo da classe não for uma pessoa, ignore-o
					if CLASSES[idx] != "person":
						continue

					# calcule as coordenadas (x, y) da caixa delimitadora
					# para o objeto
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")

					# constrói um objeto retângulo dlib a partir do limite
					# box coordena e então inicia a correlação dlib
					# rastreador
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# adicione o rastreador à nossa lista de rastreadores para que possamos
					# utilizá-lo durante pular quadros
					trackers.append(tracker)


		# caso contrário, devemos utilizar nossos objetos * rastreadores * em vez de
		# objeto * detectores * para obter uma maior taxa de processamento de quadros
		else:

			# loop sobre os rastreadores
			for tracker in trackers:
				# definir o status do nosso sistema para 'rastreamento' em vez
				# do que 'esperando' ou 'detectando'
				status = "Rastreamento"

				# atualize o rastreador e pegue a posição atualizada
				tracker.update(rgb)
				pos = tracker.get_position()

				# desempacote o objeto de posição
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# adicione as coordenadas da caixa delimitadora à lista de retângulos
				rects.append((startX, startY, endX, endY))

		# desenhe uma linha horizontal no centro do quadro - uma vez que
		# objeto cruza esta linha, determinaremos se eles foram
		# movendo 'para cima' ou 'para baixo'
		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
		cv2.putText(frame, "-Linha da Previsão - Entrada-", (10, H - ((i * 20) + 200)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		# use o rastreador de centróide para associar o (1) objeto antigo
		# centróides com (2) os centróides de objeto recém-computados
		objects = ct.update(rects)

		# loop sobre os objetos rastreados
		for (objectID, centroid) in objects.items():
			# verifique se existe um objeto rastreável para o atual
			# ID do objeto
			to = trackableObjects.get(objectID, None)

			# se não houver nenhum objeto rastreável existente, crie um
			if to is None:
				to = TrackableObject(objectID, centroid)

			# caso contrário, há um objeto rastreável para que possamos utilizá-lo
			# para determinar a direção
			else:
				# a diferença entre a coordenada y da * atual *
				# centróide e a média dos * anteriores * centróides dirão
				# em que direção o objeto está se movendo (negativo para 'para cima' e positivo para 'para baixo')
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# verifique se o objeto foi contado ou não
				if not to.counted:

					# se a direção for negativa (indicando o objeto
					# está se movendo para cima) E o centróide está acima do centro
					# linha, conte o objeto
					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						empty.append(totalUp)
						to.counted = True
					# se a direção for positiva (indicando o objeto
					# está se movendo para baixo) E o centróide está abaixo do
					# linha central, conte o objeto
					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						empty1.append(totalDown)
						#print(empty1[-1])
						x = []
						# calcula a soma do total de pessoas dentro
						x.append(len(empty1)-len(empty))
						#print("Total people inside:", x)
						# Otimize o número abaixo: 10, 50, 100, etc., indique o máx. pessoas dentro do limite
						# se o limite exceder, envie um alerta por email
						people_limit = 10
						dados.child().update({"entrada": totalDown})
						dados.child().update({"saida": totalUp})
						
						if sum(x) == people_limit:
							if config.ALERT:
								print("[INFO] Sending email alert..")
								Mailer().send(config.MAIL)
								print("[INFO] Alert sent")

						to.counted = True

			# armazene o objeto rastreável em nosso dicionário
			trackableObjects[objectID] = to

			# desenha o ID do objeto e o centroide do
			# objeto no quadro de saída
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		#constrói as informações que serão exibidas no display
		info = [
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
		]

		info2 = [
		("Total no momento", x),
		]

                # Exibir Saída
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		# Inicie um registro simples para salvar os dados no final do dia
		if config.Log:
			datetimee = [datetime.datetime.now()]
			d = [datetimee, empty1, empty, x]
			export_data = zip_longest(*d, fillvalue = '')

			with open('Log.csv', 'w', newline='') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(("End Time", "In", "Out", "Total Inside"))
				wr.writerows(export_data)


		# show the output frame
		cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
		key = cv2.waitKey(1) & 0xFF

		# se a tecla `q` foi pressionada, interrompa o loop
		if key == ord("q"):
			break

		# incrementa o número total de frames processados ​​até agora e
		# então atualize o contador FPS
		totalFrames += 1
		fps.update()

		if config.Timer:
			# Temporizador automático para parar a transmissão ao vivo. Defina para 8 horas (28800s).
			t1 = time.time()
			num_seconds=(t1-t0)
			if num_seconds > 28800:
				break

	# parar o cronômetro e exibir informações FPS
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# # se não estivermos usando um arquivo de vídeo, pare o stream de vídeo da câmera
	# se não for args.get ("input", False):
	# vs.stop ()
	#
	# # caso contrário, solte o ponteiro do arquivo de vídeo
	# outro:
	# vs.release ()

	# feche todas as janelas abertas
	cv2.destroyAllWindows()


## saiba mais sobre as diferentes programações aqui: https://pypi.org/project/schedule/

if config.Scheduler:
	## Corre a cada 1 segundo
	# schedule.every (1) .seconds.do (run)
	## Executa todos os dias (09:00). Você pode mudar isso.
	schedule.every().day.at("9:00").do(run)

	while 1:
		schedule.run_pending()

else:
	run()
