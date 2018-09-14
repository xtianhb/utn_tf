import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys

def xml_to_csv(path):
    xml_list = []
    n=0
    ListaXml=glob.glob(path + '/*.xml')
    ListaXml.sort()
    for xml_file in ListaXml:
        Nombre = os.path.split(xml_file)[1]
        NombreJpg = Nombre.split(".")[0] + ".jpg"
        if not os.path.exists(path+"/"+Nombre):
            print("No esta la imagen " + path+"/"+NombreJpg)
            continue
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
        n+=1
        print("Imagen #%04d: %s" % ( n, Nombre) )
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():

    if not os.path.exists(sys.argv[1]):
        print("No existe el path " + sys.argv[1])
        exit(-1)
	
    image_path = sys.argv[1]
    lugar, nombre = os.path.split(image_path)
	
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(image_path+"/"+nombre+".csv", index=None)
    print("Se genero el archivo " + nombre + ".csv")


main()
