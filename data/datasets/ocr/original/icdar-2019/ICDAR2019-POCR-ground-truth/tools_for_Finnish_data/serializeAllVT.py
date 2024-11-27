import os
import utilsAlto as AT
import inspect
import os
import io
import xml.etree.ElementTree as ET
import subprocess
import json
import hashlib
import re, string, timeit
import unicodedata
import epub
import csv
import gzip
import shutil
import zipfile

from multiprocessing.pool import ThreadPool
from collections import OrderedDict

def serializeAllVT(paths, vtFilesDico, keepBackLine=False, forceErase=False):
    """
    This module serialize all the VT into txt files
    """
    print("###====> RUN " + inspect.stack()[0][3] )

    # Collect paths
    baseVtPath = paths['Dir']['baseVt']
    outputDir = os.path.join(paths['Dir']['outputs'], "serialised")
    AT.ensure_dir(outputDir)

    sourceDir = os.path.join(paths['Dir']['outputs'], "source")

    for docInfo in AT.iterateVtFiles(paths, vtFilesDico, verboseLvl=2):

        i =  docInfo["docId"]
        vtCollection = docInfo["vtCollection"]
        vtInfos = docInfo["vtInfos"]

        #print (basePath + vtInfos['vt_path'] + " => " + TEMP_Dir + vtInfos['vt_txtExtract'])
        #print(os.path.join(baseVtPath, vtCollection['path']))

        for mode in ["vt", "ocr"]:

            # Use local source dir
            inPath = os.path.join(sourceDir, mode , ("#%06d_" % i) + vtCollection['name'] +  "_" + os.path.basename(vtInfos[mode+'_path']))

            # use HDD dir...
            if not os.path.exists(inPath):
                inPath = os.path.join(baseVtPath, vtCollection['path'], vtInfos[mode+'_path'])

            #if not os.path.exists(TEMP_Dir + vtInfos['vt_txtExtract']):
            serializeFile(paths=paths,
                          docInfo=docInfo,
                          inputPath = inPath,
                          outputJsonPath = os.path.join(outputDir, vtInfos[mode+'_txtExtract']),
                          keepBackLine=keepBackLine,
                          forceErase=forceErase
                          )




def serializeFile(paths, docInfo, inputPath, outputJsonPath, keepBackLine=False, forceErase=False):

    if os.path.exists(outputJsonPath) and not forceErase:
        return

    print ("SERIALIZE : " + inputPath + " => " + outputJsonPath)

    docContentJson = []

    if ".txt" in inputPath[-8:].lower():
        # It's probalby a transcription, so remove <IMAGE>
        vtContent = AT.file_get_contents(inputPath)
        vtContent = vtContent.replace("<IMAGE>", "")
        docContentJson = AT.rawtxtToJson(AT.santiseTxt(vtContent, keepBackLine=keepBackLine))


    if ".epub" in inputPath[-6:].lower() :
        bookTxt = ""
        book = AT.epub.open_epub(inputPath)
        for id, linear in book.opf.spine.itemrefs:
            data = book.read_item(book.get_item(id)).decode("utf-8")
            bookTxt += ''.join(ET.fromstring(data).itertext())
        docContentJson = AT.rawtxtToJson(AT.santiseTxt(bookTxt, keepBackLine=keepBackLine))


    elif ".json" in inputPath[-10:].lower():
        docContentJson = AT.loadDicoFromJSON(inputPath)

    elif ".xml" in inputPath[-10:].lower():

        xmlFileContent = AT.file_get_contents(inputPath)
        xmlTree, nsKey = AT.XML_parse(xmlFileContent)

        if ("alto" in nsKey):
            if "NoNamespace" in nsKey:
                nsKey = ""

            docContentJson = AT.alto_text(xmlTree, nsKey, [])

        elif "PAGE" in nsKey:
            inTmpPath = os.path.join(os.getcwd(), paths['Dir']['outputs'], "tmpXmlGz.xml")
            AT.writeUTF8File(inTmpPath, xmlFileContent)

            outTmpPath = os.path.join(os.getcwd(), paths['Dir']['outputs'], "tmpXmlGz.txt")

            pathExec = '"' + paths['Tools']['PrimaExtractorExporter_Exe'] + '"' + \
             ' -export text' + \
             ' -page-content "' + inTmpPath + '"' + \
             ' -output-folder "' + os.path.dirname(outTmpPath) + '"'

            proc = subprocess.Popen(pathExec, shell=True, stdout=subprocess.PIPE)
            proc.wait()

            pageContent = AT.file_get_contents(outTmpPath)
            docContentJson = AT.rawtxtToJson(AT.santiseTxt(pageContent, keepBackLine=keepBackLine))

        if "unknown_xml" in nsKey:
            # Supprimer toutes les balises
            vtContent = AT.santiseTxt(''.join(xmlTree.getroot().itertext()), keepBackLine=keepBackLine)
            docContentJson = AT.rawtxtToJson(AT.santiseTxt(vtContent, keepBackLine=keepBackLine))

        if "dtbook" in nsKey:
            print("ERREUR : Preprocessing needed !!!")


    newWcArray = packSerialData(docContentJson)
    serialData = {"wordList": newWcArray}
    AT.saveDicoInJSON(serialData, outputJsonPath)



def unpackSerialData(wordListArray):

    pageBounds = {}

    """
    maxK = 0
    if len(wordListArray) > 0:
        maxK = max([int(k) for k in wordListArray.keys()])
        maxK += len(wordListArray[str(maxK)]["txt"])

    #txtByteArray = bytearray((" "*maxK).encode())
    """

    txtRes = ""

    wordListArray = OrderedDict(sorted(wordListArray.items(), key=lambda t: int(t[0])))

    for k,v in wordListArray.items():

        pid = int(re.sub(u"(PAG_)|(P)", "", v["pId"]))
        intK = int(k)

        #  Rattraper si il manque des " "
        txtRes += " " * (intK - len(txtRes))

        # Ajouter le mot
        txtRes += v["txt"]

        if pid in pageBounds:
            pageBounds[pid][0] = min(pageBounds[pid][0], intK)
            pageBounds[pid][1] = max(pageBounds[pid][1], intK + len(v["txt"]))
        else:
            pageBounds[pid] = [intK, intK]

    return txtRes, pageBounds



# ICDAR 09/02/17  addBackLine=>False
def packSerialData(wcArray, addBackLine=False):

    txtSanitized = ""
    newWcArray = {}
    saniW = ""
    lastLineId = None

    # Verif
    for i in range(len(wcArray)):

        rawW = wcArray[i]["txt"]

        # ICDAR 09/02/17  addBackLine=>False
        saniW = AT.santiseTxt(rawW, keepBackLine=False).strip(" ")

        # Si on reste sur la même ligne
        if lastLineId == wcArray[i]["lId"]:
            if (len(txtSanitized) > 1) and (txtSanitized[-1] != " "):
                txtSanitized += " " # Ajouter un espace entre les mots.

        # Si on change de ligne, et que ce n'est pas la première ligne
        elif lastLineId is not None:

            # Soit on ajoute un retour à la ligne
            if addBackLine:
                if (len(txtSanitized) > 1) and (txtSanitized[-2:] != "\r\n"):
                    # Add new line
                    buf = wcArray[i].copy()
                    buf["txt"] = "\r\n"
                    buf["sId"] = "NL" + str(buf["sId"])
                    newWcArray[len(txtSanitized)] = buf
                    txtSanitized += "\r\n"

            # Soit on ajoute un simple espace
            else:
                if (len(txtSanitized) > 0) and (txtSanitized[-1] != "-") and (txtSanitized[-1] != " "):
                    # Add space if didn't finished by "-"
                    buf = wcArray[i].copy()
                    buf["txt"] = " "
                    buf["sId"] = "NL" + str(buf["sId"])
                    newWcArray[len(txtSanitized)] = buf
                    txtSanitized += " "

        wcArray[i]["txt"] = saniW
        newWcArray[len(txtSanitized)] = wcArray[i]

        txtSanitized += saniW

        lastLineId = wcArray[i]["lId"]

    return newWcArray
