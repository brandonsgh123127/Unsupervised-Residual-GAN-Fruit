import DatasetRetrieval as DataR
import ImageML as ML
import Generator as g

def main():
    #ImageProcess = DataR.DatasetRetrieval()
    #ImageProcess.retrieveImages()
    process = g.Generator()
    process.start()

main()