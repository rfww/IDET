import os

def write_path(path1,path2):
    if not os.path.exists(path2):
        os.makedirs(path2)
    files = os.listdir(path1)
    name = os.path.split(path1)[-1]+".txt"
    f = open(os.path.join(path2,name),'w')
    for file in sorted(files):
        str = os.path.join(path1, file)
        f.write(str+"\n")
    f.close()
    return True

if __name__ == "__main__":
    path1 = r'/home/wrf/2TDisk/wrf/LIVE-CD/train/label'
    path2 = r'/home/wrf/2TDisk/wrf/LIVE-CD/data_LIVE/train'
    write_path(path1, path2)
