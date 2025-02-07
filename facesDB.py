class facesDB:
    def __init__(self, source="scrape.png"):
        self.source = source
        self.faceCoordinates = {}
    
    def getEncodings(self):
        