
import pytesseract
import glob, os

from pdf2image import convert_from_path
from PIL import Image

def extractTextFromPdfs():
    pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'
    print(os.getcwd())
    for root, dirs, files in os.walk("../data"):
        for file in files:
            if file.endswith(".pdf"):
                file = os.path.join(root, file)
                print(file)
                # Store all the pages of the PDF in a variable
                pages = convert_from_path(file, 500)
                # Counter to store images of each page of PDF to image
                image_counter = 1

                # Iterate through all the pages stored above
                for page in pages:
                    filename =  file + "page_" + str(image_counter) + ".jpg"

                    # Save the image of the page in system
                    page.save(filename, 'JPEG')

                    # Increment the counter to update filename
                    image_counter = image_counter + 1

                    # Variable to get count of total number of pages
                    filelimit = image_counter - 1

                # Creating a text file to write the output
                if os.path.exists(file + ".txt"):
                    print("OCR already exists")
                    continue
                outfile = file + ".txt"

                # Open the file in append mode so that
                # All contents of all images are added to the same file
                f = open(outfile, "a")

                # Iterate from 1 to total number of pages
                for i in range(1, filelimit + 1):
                    # Set filename to recognize text from
                    # Again, these files will be:
                    # page_1.jpg
                    # page_2.jpg
                    # ....
                    # page_n.jpg
                    filename = file + "page_" + str(i) + ".jpg"

                    # Recognize the text as string in image using pytesserct
                    text = str(((pytesseract.image_to_string(Image.open(filename)))))

                    # The recognized text is stored in variable text
                    # Any string processing may be applied on text
                    # Here, basic formatting has been done:
                    # In many PDFs, at line ending, if a word can't
                    # be written fully, a 'hyphen' is added.
                    # The rest of the word is written in the next line
                    # Eg: This is a sample text this word here GeeksF-
                    # orGeeks is half on first line, remaining on next.
                    # To remove this, we replace every '-\n' to ''.
                    text = text.replace('-\n', '')

                    # Finally, write the processed text to the file.
                    f.write(text)

                # Close the file after writing all the text.
                f.close()

def extractTextForPrediction(filepath):
    pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'
    if filepath.endswith(".pdf"):
        pages = convert_from_path(filepath, 500)
        # Counter to store images of each page of PDF to image
        image_counter = 1

        # Iterate through all the pages stored above
        for page in pages:
            filename = filepath + "page_" + str(image_counter) + ".jpg"

            # Save the image of the page in system
            page.save(filename, 'JPEG')

            # Increment the counter to update filename
            image_counter = image_counter + 1

            # Variable to get count of total number of pages
            filelimit = image_counter - 1

        ocr_text = ""

        for i in range(1, filelimit + 1):
            # Set filename to recognize text from
            # Again, these files will be:
            # page_1.jpg
            # page_2.jpg
            # ....
            # page_n.jpg
            filename = filepath + "page_" + str(i) + ".jpg"

            # Recognize the text as string in image using pytesserct
            text = str(((pytesseract.image_to_string(Image.open(filename)))))

            # The recognized text is stored in variable text
            # Any string processing may be applied on text
            # Here, basic formatting has been done:
            # In many PDFs, at line ending, if a word can't
            # be written fully, a 'hyphen' is added.
            # The rest of the word is written in the next line
            # Eg: This is a sample text this word here GeeksF-
            # orGeeks is half on first line, remaining on next.
            # To remove this, we replace every '-\n' to ''.
            text = text.replace('-\n', '')

            # Finally, write the processed text to the file.
            ocr_text = ocr_text + text

        # Close the file after writing all the text.
        return ocr_text

if __name__== "__main__":
  extractTextFromPdfs()
