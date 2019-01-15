

if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join, isdir
    base_dir = 'train/audio/'
    filenames = listdir(base_dir)
    def getname(f):
        index = f.rfind('/')
        index2 = f.rfind('\\')
        if index2>index:
            index = index2
        return f[index+1:]
    files = [getname(f) for f in filenames if isdir(join(base_dir, f))]
    print(','.join(files))


