import os
import PyPDF2
import nltk
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import numpy as np
import math


def pdf2txt(r, name):
	
	if "pdf" in name:
		pdfFileObj = open(os.path.join(r, name), "rb")
		#print(r + name)
		pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
		mytext = ""
		for pageNum in range(pdfReader.numPages):
			pageObj = pdfReader.getPage(pageNum)
			mytext += pageObj.extractText()
		
		filetxt = open(os.path.join(r,name.replace("pdf","txt")), "w")
		filetxt.write(mytext)
		pdfFileObj.close()

def extraer_txt(filename):
	file = open(filename, 'rt')
	text = file.read()
	text = text.lower()
	file.close()
	return text

def limpia_texto(text):
	# dividir en palabras
	tokens = word_tokenize(text)
	#print(tokens)
	
	#Eliminar signos de Putuacion
	words = [word for word in tokens if word.isalpha()]
	#print(words)
	
	#quitar articulos
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if not w in stop_words]
	#print(words)
	
	#reducir cada palabra a su raiz o base
	porter = PorterStemmer()
	stemmed = [porter.stem(word) for word in words]
	#print(stemmed)

	cleaned_text = ""
	while len(stemmed) != 0:
		cleaned_text = cleaned_text + stemmed[0] + " "
		stemmed.pop(0)
	
	return cleaned_text 

def leerDocumentos():
	listaDoc = []
	labels   = []
	labels_doc = []

	for r, dirs, files in os.walk(os.path.join(os.getcwd(), 'training')):
		for file in files:
			if ".txt" in file:
				pdf2txt(r, file)
				text = extraer_txt(os.path.join(r,file.replace("pdf","txt")))
				cleanS = limpia_texto(text)
				#print(cleanS)
				listaDoc.append(cleanS)
				labels.append(r.replace(os.path.join(os.getcwd(), 'training'),''))
				labels_doc.append(file)
	
	return dict([('docs', listaDoc), ('labels', labels), ('labels_doc', labels_doc)])


def agregarDoc():
	test = read_all_documents()
	X_test = tfid.transform(test['docs'])
	y_test = test['labels']
	pred = clasif.predict(X_test)


def buscador(toBuscar):
	busqueda = vectorizer.transform(toBuscar)
	prediccion = clasificador.predict(busqueda)
	return (prediccion)

def extraer_docylabels(diccionario, categoria):
	
	dic_doc = {
		'documents' : [],
		'label_docs' : []
	}

	for i, cluster_name in enumerate(diccionario['labels']):
		if cluster_name == categoria:
			dic_doc['documents'].append(diccionario['docs'][i])
			dic_doc['label_docs'].append(diccionario['labels_doc'][i])			

	return dic_doc

def kmeans_busq(dic_doc, consulta, cant):

	docs = []
	for i in range(cant):
		vectorizer = TfidfVectorizer()
		vectorizer.fit(dic_doc['documents'])
		vector = vectorizer.transform(dic_doc['documents'])
		labels_doc = dic_doc['label_docs']	
		tam = len(labels_doc)
		clasif_doc = KMeans(tam) #predice la categoria de un doc
		clasif_doc.fit(vector, labels_doc)
		try: 
			doc = labels_doc[np.where(consulta == clasif_doc.labels_)[0][0]]
			docs.append(doc)
			ind = dic_doc['label_docs'].index(doc)
			dic_doc['label_docs'].pop(ind)
			dic_doc['documents'].pop(ind)
		except IndexError:
			print("\n Ops! Solo se consiguieron estos documentos")
			print("\n***************************************************")
			break
	return docs

def imprimir_doc(docs, categoria):

	DIR = os.path.join(os.getcwd(),'training')
	d = os.path.join(DIR,categoria.replace('/', ''))
	directorios = os.listdir(d)
	for doc in docs:
		nomb_doc = doc.replace("txt","pdf")
		print("\nSe encuentra en: ", nomb_doc , " dentro de")
		print(d)
		print("***************************************************\n")
		ind = directorios.index(doc)
		#print(extraer_txt(os.path.join(d, directorios[ind])))
		mostrar_pdf(os.path.join(d, directorios[ind]))
		
def mostrar_pdf(name):

	file_pdf = open(name.replace("txt","pdf"), 'rb')
	pdfReader = PyPDF2.PdfFileReader(file_pdf)	
	
	for pageNum in range(pdfReader.numPages):
		pageObj = pdfReader.getPage(pageNum)
		print(pageObj.extractText())



		
# Se crea la lista de etiquetas para su clasificacion

def obtener_cantdoc(cat, labels):
	cont = 0
	for etiq in labels:
		if etiq == cat:
			cont += 1
	return cont


dic = leerDocumentos()
""" https://unipython.com/como-preparar-datos-de-texto-con-scikit-learn/ """
# crear la transformación
vectorizer = TfidfVectorizer()
# tokenizar y construir vocabulario
vectorizer.fit(dic['docs'])
# resumir
#print(vectorizer.vocabulary_)
#print(vectorizer.idf_)
# documento codificado
vector = vectorizer.transform(dic['docs'])
# resumir vector codificado
#print(vector.shape)
#print(vector.toarray())
labels = dic['labels']
#Aplicacion de K-means que usa tambien la distancia Euclideana
clasificador = KMeans(3) #predice la categoria de un doc
clasificador.fit(vector, labels)
print("\n***************************************************")
print("\n*****  Buscar doc. segun lo solicitado          ***")
print("\n***************************************************")
busq = input("\nBuscar: ")
b = []
b.append(busq)
prediccion = buscador(b)
categoria = labels[np.where(prediccion == clasificador.labels_)[0][0]]
cant = math.ceil(obtener_cantdoc(categoria, labels)/2)
cant_doc = input("Cant. de doc(Max  " + str(cant) + "): ")
print("\n***************************************************")
diccionario = extraer_docylabels(dic, categoria)
doc_encontrados = kmeans_busq(diccionario, prediccion, cant)
imprimir_doc(doc_encontrados, categoria)


