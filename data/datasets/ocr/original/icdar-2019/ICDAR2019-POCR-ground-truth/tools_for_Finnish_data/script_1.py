import csv, os, re, numpy
import alignment
from xml.dom.minidom import parse

inputFile = "input"+os.sep+"nlf_ocr_gt_tescomb5_2017.csv"
outputFileFolderPath = "output"+os.sep+"full"+os.sep

writeInFile = True
verbose = False

def writeUTF8File(path, text):

    if ".gz" in path[-4:].lower():
        with gzip.open(path, "wb") as f:
            f.write( text.encode("utf-8-sig") )
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

def countChar(itemTXT):
    
    nbCH = 0
    nbWD = 0
    
    #itemTXT = str.split("[OCR_aligned]")[0][13:]
    for wd in itemTXT.split(' '):
        nbWD += 1
        nbCH += len(wd)
        
    return nbCH, nbWD


with open(inputFile, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    done = []
    nbTL = 0
    nbCH = 0
    nbWD = 0
    nbER = 0
    
    #Init global vars
    nbTLG = 0
    nbWDG = 0
    nbCHG = 0
    nbERG = 0
    #scores = []
    #CERs = []
    
    # Init vars
    txtIn = ""
    txtAl = ""
    txtGt = ""
    
    n = 0
    
    nbFile = "500644"
    pageName = ""
    doProcess = True
    
    for i,row in enumerate(spamreader):
                    
        #Skip headers
        if i < 4:
            continue
            
        #Build output path
        outputFileDataset = outputFileFolderPath + str(n) + '.txt'
            
        
        #Get page names
        lastPageName = pageName
        pageName = row[3]
        
        #If next page name is different from the current page then save the previous data and init vars
        if pageName != lastPageName and i > 4:
                
            #OPTION: skip too big files
            if (len(txtIn) or len(txtGt)) > 15000:
                print('['+str(n+1)+'/'+str(nbFile)+'] SKIP alignment because '+str(max(len(txtIn),len(txtGt)))+' > 15000 char')
                #Add current page to done list
                done.append(pageName)
                #Add # ignored items
                nbERG += len(txtGt)
                # Init vars
                txtIn = ""
                txtAl = ""
                txtGt = ""
                n += 1
                continue

            
            #######################
            # Build [OCR_aligned] #
            #######################
            
            if (len(txtIn) or len(txtGt)) > 15000:
                print('['+str(i)+'/'+str(nbFile)+']SKIP alignment because > 15000 char')
                n += 1
                continue
            
            print('Running alignment.needle('+str(len(txtIn))+','+str(len(txtGt))+')...')
            
            try:
                #ocrRes, vtRes, scoreMatch = alignment.needle(txtIn, replaceChar(txtGt))
                ocrRes, vtRes, scoreMatch = alignment.needle(txtIn, txtGt)
            except:
                print('\nERROR in alignment.needle() for',fi)
                continue
                #print('\n['+str(i)+'/'+str(nbFile)+'] Running alignment.needle('+str(len(txtIn))+','+str(len(txtGt))+')...')

                scoreMatch = numpy.around( scoreMatch, decimals=2 )

            #Stats
            for wd in vtRes.split(' '):
                nbWD += 1
                nbCH += len(wd)
            #nbTLG += nbTL
            nbWDG += nbWD
            nbCHG += nbCH
            #scores.append(scoreMatchED)
            #CERs.append(CER)
        
            #Display        
            print('['+str(i)+'/'+str(nbFile)+']('+str(i+1)+') '+str(scoreMatch)+'% match',len(txtIn),'[OCR_toInput]','/',len(txtGt),'[ GS_aligned] from',lastPageName)
            #print('\n',key)

            if verbose or scoreMatch <= 5:# or fo == "BSB_DE":
                printmd("**[OCR_toInput]**")
                print(txtIn)
                printmd("**[OCR_aligned]**")
                print(ocrRes)
                printmd("**[ GS_aligned]**")
                print(vtRes)
                print()

            #Write to file
            if writeInFile:

                #Build outputFileDataset path
                writeUTF8File(outputFileDataset,
                 "[OCR_toInput] " + txtIn + "\n" +
                 "[OCR_aligned] " + ocrRes+ "\n" +
                 "[ GS_aligned] " + vtRes + "\n"
                )
                
            # Init vars
            txtIn = ""
            txtAl = ""
            txtGt = ""

            #Increment file ID
            n += 1
        
        #Get GT + OCR output text from CSV file
        
        #Add a space if already contain some text
        if len(txtIn) > 0:
            sep = ' '
        else:
            sep = ''

        #Get FineReder 11 OCR output (row[6] = Tesseract)
        txtIn = txtIn + sep + row[8]
        txtGt = txtGt + sep + row[5]
              

    #Show sum stats for the entire folder
    print('\n\n',nbTLG,'texlines,',nbWDG,'words',nbCHG,'characters including',nbERG,'ignored chracters from GT')
