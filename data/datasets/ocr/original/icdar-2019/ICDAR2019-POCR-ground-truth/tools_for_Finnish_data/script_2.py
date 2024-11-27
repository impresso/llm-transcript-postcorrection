import os
from shutil import copyfile

#Functions
def getFinalText(path):
    """
        Load OCR output, OCR aligned and GT aligned from output file for stats
    """
    
    with open(path, "r") as f:
        #print(f.encoding)
        txt = f.read();
        split = txt.split('\n')
        if len(split) == 4:
            txtIn  = split[0]
            ocrRes = split[1]
            vtRes  = split[2]
            
            #_, _, scoreMatch = alignment.finalize(txtIn, vtRes, False)
            #print(txtIn)
            #raise
    
            return txtIn, ocrRes, vtRes#, scoreMatch
        else:
            return None
    
#Main

#Load commun methods firstµ

inPath  = 'output/full/'
outTrainPath = 'output/training/'
outTestPath = 'output/evaluation/'

#Nb character per document sets
charDic = {
    "FI1":1568276
}

#for langDir in os.listdir(os.path.join(inPath,lang)):

langDir = "FI1"

#Show current sub directory name
print('\n',langDir)

nbWD = nbWD1 = nbWD2 = 0
nbCH = nbCH1 = nbCH2 = 0
iTrain = iTest = 0

nbMax = charDic[langDir]
for file in sorted(os.listdir(inPath)):
    if file[0] == '.':
        print('SKIP',file)
        continue
    
    #Load text
    txtIn, ocrRes, vtRes = getFinalText(os.path.join(inPath,file))
                    
    if nbCH < nbMax:
        
        #Copy train file
        filename = str(iTrain)+'.txt'
        copyfile(os.path.join(inPath,file), os.path.join(outTrainPath,filename))
        
        #Increment
        iTrain += 1
        
        #Count training character
        for wd in vtRes.split(' '):
            #nbWD += 1
            nbCH += len(wd)
        
        print('.',end='')
    else:
        
        #Copy test file
        filename = str(iTest)+'.txt'
        copyfile(os.path.join(inPath,file), os.path.join(outTestPath,filename))
        
        #Increment
        iTest += 1
        
        #Count testing character
        for wd in vtRes.split(' '):
            #nbWD1 += 1
            nbCH1 += len(wd)
            
        print('-',end='')
        
    #Count all character
    for wd in vtRes.split(' '):
        #nbWD2 += 1
        nbCH2 += len(wd)

print('\n',nbCH,'+',nbCH1,'=',nbCH2)