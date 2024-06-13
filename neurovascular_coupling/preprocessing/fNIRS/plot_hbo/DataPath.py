import os
from queue import LifoQueue

class DataPath:
    
    def __init__(self, baseline_path: str, fif: bool=False,  recursive: bool=True) -> None:
        self.stack = LifoQueue(maxsize=100)
        self.data_path = list()
        self.baseline_path = baseline_path
        self.stack.put(self.baseline_path)
        self.iter = 0
        self.isFif = fif
        if recursive:
            self.recurrentDirSearch()
        else:
            self.getAllinOneDir()
    
    def getAllinOneDir(self):
        onlyfiles = self.get_immediate_files(self.baseline_path)
        for file in onlyfiles:
            if file.find(".snirf") != -1 or file.find(".nirf") != -1: 
                self.data_path.append(os.path.join(self.baseline_path,file))

    def get_immediate_subdirectories(self, a_dir):
        return [os.path.join(a_dir, name) for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
    
    def get_immediate_files(self, a_dir):
        return [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]

    def isThisTheFinalDir(self, a_dir):
        onlyfiles = self.get_immediate_files(a_dir)
        for file in onlyfiles:
            if not self.isFif:
                if file.find(".snirf") != -1 or file.find(".nirf") != -1:
                    return os.path.join(a_dir,file)
            elif self.isFif:
                if file.find(".fif") != -1:
                    return os.path.join(a_dir,file)
        return None
    
    def recurrentDirSearch(self):
        self.iter += 1 
        if self.stack.empty():
            return self.data_path
        else:
            a_dir = self.stack.get()
            file = self.isThisTheFinalDir(a_dir)
            if file is not None:
                self.data_path.append(file)
            else:
                subDirs = self.get_immediate_subdirectories(a_dir)
                if subDirs is not None:
                    for dir in subDirs:
                        self.stack.put(dir)
            return self.recurrentDirSearch()
        
    def getDataPaths(self):
        return self.data_path