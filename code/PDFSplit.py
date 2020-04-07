from PyPDF2 import PdfFileWriter, PdfFileReader
import glob, os

os.chdir("../temp/source")
for file in glob.glob("*.pdf"):

    inputpdf = PdfFileReader(open(file, "rb"))

    for i in range(inputpdf.numPages):
        output = PdfFileWriter()
        output.addPage(inputpdf.getPage(i))
        
        with open(file + "%s.pdf" % i, "wb") as outputStream:
            output.write(outputStream)
