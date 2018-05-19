import random
from os import listdir
from PIL import Image
from datetime import datetime

trainInputImagesPath = 'airs-dataset/train-input'
trainOutputImagesPath = 'airs-dataset/train-output'
testInputImagesPath = 'airs-dataset/test-input'
testOutputImagesPath = 'airs-dataset/test-output'
validInputImagesPath = 'airs-dataset/valid-input'
validOutputImagesPath = 'airs-dataset/valid-output'

trainInputImagesFiles = listdir(trainInputImagesPath)
trainOutputImagesFiles = listdir(trainOutputImagesPath)
testInputImagesFiles = listdir(testInputImagesPath)
testOutputImagesFiles = listdir(testOutputImagesPath)
validInputImagesFiles = listdir(validInputImagesPath)
validOutputImagesFiles = listdir(validOutputImagesPath)


print(str(datetime.now()) + ': trainInputImagesFiles:', len(trainInputImagesFiles))
print(str(datetime.now()) + ': trainOutputImagesFiles:',  len(trainOutputImagesFiles))
if(len(trainInputImagesFiles) != len(trainOutputImagesFiles)):
    raise Exception('train input images and output images number mismatch')

print(str(datetime.now()) + ': testInputImagesFiles:', len(testInputImagesFiles))
print(str(datetime.now()) + ': testOutputImagesFiles:', len(testOutputImagesFiles))
if(len(testInputImagesFiles) != len(testOutputImagesFiles)):
    raise Exception('test input images and output images number mismatch')

print(str(datetime.now()) + ': validInputImagesFiles:', len(validInputImagesFiles))
print(str(datetime.now()) + ': validOutputImagesFiles:', len(validOutputImagesFiles))
if(len(validInputImagesFiles) != len(validOutputImagesFiles)):
    raise Exception('valid input images and output images number mismatch')

for i in range(len(trainInputImagesFiles)):
    inputImageFile = trainInputImagesFiles[i][:-5]
    outputImageFile = trainOutputImagesFiles[i][:-4]
    if(inputImageFile != outputImageFile):
        raise Exception('train inputImageFile and outputImageFile mismatch at index', str(i))

for i in range(len(testInputImagesFiles)):
    inputImageFile = testInputImagesFiles[i][:-5]
    outputImageFile = testOutputImagesFiles[i][:-4]
    if(inputImageFile != outputImageFile):
        raise Exception('test inputImageFile and outputImageFile mismatch at index', str(i))
        
for i in range(len(validInputImagesFiles)):
    inputImageFile = validInputImagesFiles[i][:-5]
    outputImageFile = validOutputImagesFiles[i][:-4]
    if(inputImageFile != outputImageFile):
        raise Exception('valid inputImageFile and outputImageFile mismatch at index', str(i))

print(str(datetime.now()) + ': input and output files check success')


def writeDataFile(inputImagePath, outputImagePath, inputImageFiles, outputImageFiles, dataFileName):
    dataFile = open(dataFileName, 'w')
    rectSize = 5
    linesCount = 0
    linesLimit = 200000
    linesCountPerImage = 0
    linesLimitPerImage = (linesLimit / len(inputImageFiles)) + 1
    
    
    for i in range(len(inputImageFiles)):
        print(str(datetime.now()) + ': prcessing image', i)
        linesCountPerImage = 0
        inputImage = Image.open(inputImagePath + '/' + inputImageFiles[i])
        inputImageXSize, inputImageYSize = inputImage.size
        inputImagePixels = inputImage.load()
        
        outputImage = Image.open(outputImagePath + '/' + outputImageFiles[i])
        outputImageXSize, outputImageYSize = outputImage.size
        outputImagePixels = outputImage.load()
        
        if((inputImageXSize != outputImageXSize) or (inputImageYSize != outputImageYSize)):
            raise Exception('train inputImage and outputImage mismatch at index', str(i)) 
        
        outputImageBldgPixelsArr = [];
        outputImageNonBldgPixelsArr= [];
        
        
        
        for x in range(rectSize//2, inputImageXSize - (rectSize//2)):
            for y in range(rectSize//2, inputImageYSize - (rectSize//2)):
                isBldgPixel = outputImagePixels[x, y]
                
                if(isBldgPixel[0] == 255):
                    outputImageBldgPixelsArr.append((x, y))
                else:
                    outputImageNonBldgPixelsArr.append((x, y))
        
       
        random.shuffle(outputImageBldgPixelsArr)
        random.shuffle(outputImageNonBldgPixelsArr)
    
        for m in range(2000):
            
            if(linesCountPerImage >= linesLimitPerImage):
                break
            
            #f((m) >= len(outputImageNonRoadPixelsArr)):
                #break
            
            x = outputImageBldgPixelsArr[m][0];
            y = outputImageBldgPixelsArr[m][1];
            
            #rect = (x - (rectSize//2), y - (rectSize//2), x + (rectSize//2) + 1, y + (rectSize//2) + 1)
            #subImage = inputImage.crop(rect).load()
            line = ''
            
            for i in range(x - (rectSize//2),x + (rectSize//2) + 1):
                for j in range(y - (rectSize//2),y + (rectSize//2) + 1):
                    line += str(inputImagePixels[i, j][0]) + ','
                    line += str(inputImagePixels[i, j][1]) + ','
                    line += str(inputImagePixels[i, j][2]) + ','
                    
            
            line += str(1) + '\n'
            linesCount += 1
            linesCountPerImage += 1
            dataFile.write(line)
            
            
            x = outputImageNonBldgPixelsArr[m][0];
            y = outputImageNonBldgPixelsArr[m][1];
                
                #rect = (x - (rectSize//2), y - (rectSize//2), x + (rectSize//2) + 1, y + (rectSize//2) + 1)
                #subImage = inputImage.crop(rect).load()
            line = ''
            for i in range(x - (rectSize//2),x + (rectSize//2) + 1):
                for j in range(y - (rectSize//2),y + (rectSize//2) + 1):
                    line += str(inputImagePixels[i, j][0]) + ','
                    line += str(inputImagePixels[i, j][1]) + ','
                    line += str(inputImagePixels[i, j][2]) + ','
                
            line += str(0) + '\n'
            linesCount += 1
            linesCountPerImage += 1
            dataFile.write(line)
    
    print(str(datetime.now()) + ': ' + dataFileName + ' linesCount:', linesCount)

trainDataFileName = 'airs-dataset/train.csv'
testDataFileName = 'airs-dataset/test.csv'
validDataFileName = 'airs-dataset/valid.csv'

print(str(datetime.now()) + ': writing trainDataFile')
writeDataFile(trainInputImagesPath, trainOutputImagesPath, trainInputImagesFiles, trainOutputImagesFiles, trainDataFileName)
print(str(datetime.now()) + ': trainDataFile complete')

print(str(datetime.now()) + ': writing testDataFile')
writeDataFile(testInputImagesPath, testOutputImagesPath, testInputImagesFiles, testOutputImagesFiles, testDataFileName)
print(str(datetime.now()) + ': testDataFile complete')

print(str(datetime.now()) + ': writing validDataFile')
writeDataFile(validInputImagesPath, validOutputImagesPath, validInputImagesFiles, validOutputImagesFiles, validDataFileName)
print(str(datetime.now()) + ': validDataFile complete')
