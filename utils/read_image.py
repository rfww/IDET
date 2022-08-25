
def read_image(path):
    image = []
    with open(path, 'r') as f:
        for line in f:
            image.append(line.strip())
    f.close()
    return image



def read_label(path):
    label = []
    with open(path, 'r') as f:
        for line in f:
            label.append(line.strip())
    f.close()
    return label
