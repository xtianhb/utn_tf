import cv2
import os
import shutil
import glob
import sys
import numpy as np
import random
import xml.etree.ElementTree as ET
########################################################################
#Lee una carpeta con fotos recortadas y genera dataset con fotos incrustadas en fondo
#cd path/to/annotating
#python gen_set.py Src Dest 10
########################################################################
View=0  #Ver procesos
NxFoto=10  #Cantidad por foto
WIDTH_CANVAS=640  #ancho canvas
HEIGHT_CANVAS=480 #alto canvas
HEIGHT_OBJ_MIN=int(HEIGHT_CANVAS//10) #Alto foto
HEIGHT_OBJ_MAX=int(HEIGHT_CANVAS//5) #Alto foto
Ang=5 #variacion de angulo + y -
Backg=[]
NBack=0
########################################################################
def IndentXml(elem, level=0, hor='\t', ver='\n'):
	i = ver + level * hor
	if len(elem):
		if not elem.text or not elem.text.strip():
			elem.text = i + hor
		if not elem.tail or not elem.tail.strip():
			elem.tail = i
		for elem in elem:
			IndentXml(elem, level + 1, hor, ver)
		if not elem.tail or not elem.tail.strip():
			elem.tail = i
	else:
		if level and (not elem.tail or not elem.tail.strip()):
			elem.tail = i
	return
########################################################################
def Add_SubEl(Tree, Name="name", Val=""):
	NE = ET.SubElement(Tree, Name)
	NE.text = Val
	return
########################################################################
def Rotar(Img, Ang):
	H,W = Img.shape[0:2] # shape(w,h,3) Obtiene W y H antes de procesar
	cX = W//2
	cY = H//2
	M = cv2.getRotationMatrix2D( (cX, cY), Ang, 1.0) #Genera datos de rotacion
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	nW = int((H * sin) + (W * cos))
	nH = int((H * cos) + (W * sin))
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY
	Img = cv2.warpAffine( Img, M, (nW,nH) )
	return Img
########################################################################
def AddCanvas(Img):
	
	Ho,Wo = Img.shape[0:2] # shape(w,h,3) Obtiene W y H antes de procesar
	#print("Image size: %dx%d" % (Wo, Ho) )
	AR = float(Wo)/float(Ho)   # W/H calcula AR  
	
	Ho = random.randint(HEIGHT_OBJ_MIN,HEIGHT_OBJ_MAX)   # nueva altura  
	Wo = int( float(Ho) * AR )  #Nuevo ancho respetando AR 
	
	Img = cv2.resize( Img, ( Wo,Ho ) )   # cv2.resize( src, (w,h) ) Escala imagen al nuevo size
	
	PathBackg = Backg[random.randint(0,NBackg-1)]

	Canvas = cv2.imread(PathBackg)
	Hc,Wc = Canvas.shape[0:2]
	ARc=Wc/Hc
	if Wc>WIDTH_CANVAS:
		Canvas = cv2.resize(Canvas, (WIDTH_CANVAS, int(WIDTH_CANVAS/ARc)  ))  #Imagen mas grande vacia, canvas
	elif Hc>HEIGHT_CANVAS:
		Canvas = cv2.resize(Canvas, (int(HEIGHT_CANVAS/ARc), HEIGHT_CANVAS  ))  #Imagen mas grande vacia, canvas
		
	Hc,Wc = Canvas.shape[0:2]
	Ox=random.randint(1,Wc-Wo-5)  #offset x
	Oy=random.randint(1,Hc-Ho-5)  #offset y
	Canvas[Oy:Oy+Img.shape[0], Ox:Ox+Img.shape[1]] = Img   #incrusta imagen chica en canvas
	
	return Canvas,Ho,Wo,Ox,Oy
########################################################################
def AddNoise(Img):
	W,H = Img.shape[0:2]
	Noise =  random.randint(1, 30)*np.random.randn(W, H, 3)
	Img =  Img.astype("uint8") + Noise.astype("uint8")
	return Img
########################################################################
def AddBlur(Img):
	Img=cv2.blur(Img,(3,3))
	return Img
########################################################################
def Trf_Img(ImgDestPath):

	Img = cv2.imread(ImgDestPath)
	
	Img = Rotar(Img, random.randint(-Ang, Ang))
	
	#Img = AddNoise(Img)
	
	Img,Hc,Wc,Ox,Oy = AddCanvas(Img)	
	
	Img = AddBlur(Img)
	
	if View:
		cv2.imshow("Image", Img)
		cv2.waitKey(1)
	
	cv2.imwrite(ImgDestPath, Img)
	
	Hi,Wi = Img.shape[0:2]
	
	return Wi,Hi,Wc,Hc,Ox,Oy
########################################################################
def GenXml(XmlDestPath, ImgDestPath, Wi,Hi,Wo,Ho,Ox,Oy, ClassName):
	
	if os.path.exists(XmlDestPath):
		os.remove(XmlDestPath)
		
	XmlRoot = ET.Element("annotating")
	
	Add_SubEl(XmlRoot, "folder", os.path.split(ImgDestPath)[0] )
	Add_SubEl(XmlRoot, "filename", os.path.split(ImgDestPath)[1] )
	Add_SubEl(XmlRoot, "path", ImgDestPath )
	
	Source = ET.SubElement(XmlRoot, "source")
	Add_SubEl(Source, "database", "Unknown")
	
	Size = ET.SubElement(XmlRoot, "size")
	Add_SubEl(Size, "width", str(Wi))
	Add_SubEl(Size, "height", str(Hi))
	Add_SubEl(Size, "depth", "3")
		
	Add_SubEl(XmlRoot, "segmented", "0")
	
	Obj = ET.SubElement(XmlRoot, "object")
	Add_SubEl(Obj, "name", ClassName)
	Add_SubEl(Obj, "pose", "Unspecified")
	Add_SubEl(Obj, "truncated", "0")
	Add_SubEl(Obj, "difficult", "0")	
	
	BndBox = ET.SubElement(Obj, "bndbox")
	Add_SubEl(BndBox, "xmin", str(Ox) )
	Add_SubEl(BndBox, "ymin", str(Oy) )
	Add_SubEl(BndBox, "xmax", str(Ox+Wo) )
	Add_SubEl(BndBox, "ymax", str(Oy+Ho) )
	
	IndentXml(XmlRoot)
	
	XmlTree=ET.ElementTree(XmlRoot)
	XmlTree.write(XmlDestPath)
	
	return
########################################################################
def Main_App():
	global NxFoto, NBackg, Backg
	
	print("Inicio")
	
	if len(sys.argv)<2:
		print("Falta carpeta origen")
		print("python gen_set.py /path/to/src /path/to/dest NFotos")
		exit(-1)
	
	if len(sys.argv)<3:
		print("Falta carpeta destino")
		print("python gen_set.py /path/to/src /path/to/dest NFotos")
		exit(-1)
		
	if len(sys.argv)<4:
		print("Falta # fotos por ejemplo")
		print("python gen_set.py /path/to/src /path/to/dest NFotos")
		exit(-1)
		
	if not os.path.exists(sys.argv[1]):
		print("No existe el path " + sys.argv[1])
		exit(-1)

	SourcePath = sys.argv[1]	
	DestPath = sys.argv[2]
	NxFoto = int(sys.argv[3])
	
	ScriptDir=os.path.split(os.path.realpath(__file__))[0]
	Backg=glob.glob(ScriptDir+"/backg/*.jpg")
	NBackg=len(Backg)
	if len(Backg) == 0:
		print(ScriptDir)
		print("Error no se encontraron backgs")
		exit(-1)
		
	N=0
	
	Lista = glob.glob(SourcePath + '/*.jpg')
	Lista.sort()
	for ImgFilePath in Lista:
		
		ImgName = os.path.split(ImgFilePath)[1] #Se queda con el nombre de la "foto.jpg"
		ImgName = ImgName.split(".")[0] #Se queda con el nombre de la "foto"
		ClassName = "utn"
		
		for K in range (0, NxFoto):
			sK=str(K)
			lsK=len(sK)
			ImgDestPath = DestPath+"/"+ImgName+"_"+('0'*(3-lsK))+sK+".jpg"
			XmlDestPath = DestPath+"/"+ImgName+"_"+('0'*(3-lsK))+sK+".xml"
		
			if os.path.exists(XmlDestPath):
				#print("Ya esta el xml " + XmlDestPath)
				os.remove(XmlDestPath)
				pass

			shutil.copy(ImgFilePath, ImgDestPath) # Copia template xml
			#shutil.copy(SrcTempXml, XmlDestPath) # Copia template xml
		
			Wi,Hi,Wc,Hc,Ox,Oy = Trf_Img(ImgDestPath)
			GenXml(XmlDestPath, ImgDestPath, Wi,Hi,Wc,Hc,Ox,Oy, ClassName)
		
			print("Imagen %04d %s" % (N, ImgDestPath))
			
			N+=1
		
			#if View:
			#	break
				
		#if View:
		#	break
		
	return
########################################################################
Main_App()
exit(0)
########################################################################

